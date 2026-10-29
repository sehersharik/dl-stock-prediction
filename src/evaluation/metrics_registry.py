"""Single source of truth for reported results: reports/metrics.json.

Built by `python main.py metrics` from saved artifacts only (registry runs, the frozen
archive of the previous run, the feature-selection file and Backtester re-runs). The
dashboard, the live inference engine and the README all read the run named by
`active_run`, so a number shown anywhere can be traced back to one entry here.
"""
import copy
import json
import logging
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import torch

from src.backtesting.engine import Backtester
from src.evaluation.metrics import evaluate_classification
from src.evaluation.splitter import TimeSeriesSplitter
from src.experiments.feature_ablation import get_feature_set
from src.features.generator import align_trading_calendar
from src.models.dl_models import ConfigurableMLP
from src.utils.config import PROJECT_ROOT, load_config

logger = logging.getLogger(__name__)

METRICS_PATH = PROJECT_ROOT / "reports" / "metrics.json"
TICKER = "RELIANCE.NS"
TARGET = "Target_Direction"
THRESHOLD = 0.5
OLD_RUN_ID = "20261007_003042"
OLD_ARCHIVE = Path("archive") / f"run_{OLD_RUN_ID}"
OLD_LSTM_RUN_ID = "20261007_003103"  # the 27-feature LSTM trained alongside the old MLP
# (transaction cost, slippage) charged on every position change, i.e. per side
COST_GRID = [(0.0, 0.0), (0.001, 0.0005), (0.002, 0.0005), (0.002, 0.001)]
DEFAULT_COST_INDEX = 1


def load_metrics(path: Path = METRICS_PATH) -> dict:
    with open(path) as f:
        return json.load(f)


def active_run(metrics: dict = None) -> dict:
    """The run the dashboard, README and live engine use, with its id under 'run_id'."""
    metrics = metrics or load_metrics()
    run_id = metrics["active_run"]
    return {"run_id": run_id, **metrics["runs"][run_id]}


def _rel(path: Path) -> str:
    return str(Path(path).relative_to(PROJECT_ROOT)) if Path(path).is_absolute() else str(path)


def _read_xy(proc_dir: Path):
    X = pd.read_csv(proc_dir / f"{TICKER}_X.csv", index_col=0, parse_dates=True)
    y = pd.read_csv(proc_dir / f"{TICKER}_y.csv", index_col=0, parse_dates=True)[TARGET]
    return X, y


def _split_summary(splits: dict) -> dict:
    return {name: {"start": str(Xs.index[0].date()), "end": str(Xs.index[-1].date()),
                   "n": int(len(Xs)), "up_share": float(ys.mean())}
            for name, (Xs, ys) in splits.items()}


def _evaluate_saved_mlp(run_dir: Path, X: pd.DataFrame, y: pd.Series, feature_list: list, config: dict) -> dict:
    """Score a saved MLP + scaler on a given split (used for validation metrics only)."""
    dl_cfg = config['models'].get('deep_learning', {})
    model = ConfigurableMLP(len(feature_list), dl_cfg.get('layers', [32, 16]), dl_cfg.get('dropout', 0.2))
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu"))
    model.eval()
    scaler = joblib.load(run_dir / "scaler.joblib")
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.FloatTensor(scaler.transform(X[feature_list])))).numpy().ravel()
    return evaluate_classification(y.values, (probs >= THRESHOLD).astype(int), probs)


def _backtest_grid(pred_path: Path, config: dict) -> dict:
    grid = []
    for tc, sl in COST_GRID:
        cfg = copy.deepcopy(config)
        cfg['backtesting'] = {**cfg.get('backtesting', {}), 'transaction_cost': tc, 'slippage': sl}
        m = Backtester(cfg, PROJECT_ROOT).run(TICKER, "mlp", TARGET, THRESHOLD, pred_path=pred_path, save=False)
        grid.append({"transaction_cost": tc, "slippage": sl, "cost_per_position_change": tc + sl, **m})
    return {"cost_convention": "cost + slippage charged on every position change (per side; a round trip pays twice)",
            "threshold": THRESHOLD, "default_index": DEFAULT_COST_INDEX, "cost_grid": grid}


def _baselines(models_dir: Path, names: list, feature_list: list = None) -> dict:
    out = {}
    for name in names:
        path = models_dir / f"{TICKER}_{name}_{TARGET}_metrics.json"
        if not path.exists():
            continue
        d = json.loads(path.read_text())
        if feature_list is not None and d.get("feature_list") not in (None, feature_list):
            raise ValueError(f"Baseline {name} was trained on a different feature set")
        out[name] = {split: d["metrics"][split] for split in ("train", "val", "test")}
        out[name]["source"] = _rel(path)
    return out


def _model_block(feature_list: list, config: dict) -> dict:
    dl_cfg = config['models'].get('deep_learning', {})
    return {"type": "mlp", "layers": dl_cfg.get('layers', [32, 16]), "dropout": dl_cfg.get('dropout', 0.2),
            "threshold": THRESHOLD, "n_features": len(feature_list), "feature_list": feature_list}


def _old_run_entry(config: dict) -> dict:
    """Run 20261007_003042, rebuilt from its frozen archive (pre-Phase-3 data and split)."""
    A = PROJECT_ROOT / OLD_ARCHIVE
    run_dir = PROJECT_ROOT / "models" / "RELIANCE" / "mlp" / OLD_RUN_ID
    meta = json.loads((run_dir / "metadata.json").read_text())
    X, y = _read_xy(A / "data" / "processed")
    n = len(X)
    tr, va = int(n * 0.70), int(n * 0.85)  # the 70/15/15 row split this run used
    splits = {"train": (X.iloc[:tr], y.iloc[:tr]), "val": (X.iloc[tr:va], y.iloc[tr:va]), "test": (X.iloc[va:], y.iloc[va:])}
    features = meta["feature_list"]
    ablation = json.loads((A / "reports" / f"{TICKER}_ablation_results.json").read_text())
    level_keys = ("ROC-AUC", "Accuracy", "F1", "Balanced Accuracy")
    pred_path = A / "data" / "predictions" / f"{TICKER}_mlp_{TARGET}_test_preds.csv"
    raw = pd.read_csv(PROJECT_ROOT / "data" / "raw" / f"{TICKER}_10y.csv", index_col=0)
    lstm_meta = json.loads((PROJECT_ROOT / "models" / "RELIANCE" / "lstm" / OLD_LSTM_RUN_ID / "metadata.json").read_text())
    if lstm_meta["feature_list"] != features:
        raise ValueError("Old LSTM run used a different feature set")
    return {
        "label": "Previous deployed run (pre-Phase-3 data; feature set chosen on test)",
        "model": _model_block(features, config),
        "data": {"raw_sessions": int(len(raw)), "rows": int(n),
                 "calendar_rule": "none: holiday placeholder bars kept, leaving ~50-row gaps after each",
                 "known_issues": ["368 sessions missing from the feature matrix inside its date range",
                                  "19 targets spanned more than one session (5 in the test split)"]},
        "split": {"method": "70/15/15 by row count", **_split_summary(splits)},
        "feature_selection": {"split": "test", "metric": "ROC-AUC", "selected_level": "F5",
                              "levels": {lvl: {"n_features": len(get_feature_set(features, int(lvl[1:]))), **{k: v[k] for k in level_keys}}
                                         for lvl, v in ablation["mlp"].items()},
                              "lstm_levels": {lvl: {k: v[k] for k in level_keys} for lvl, v in ablation.get("lstm", {}).items() if v}},
        "metrics": {"mlp": {"val": _evaluate_saved_mlp(run_dir, *splits["val"], features, config), "test": meta["metrics"]},
                    "baselines": _baselines(A / "models", ["logistic", "random_forest"]),
                    "lstm": {"test": lstm_meta["metrics"], "run_id": OLD_LSTM_RUN_ID,
                             "sequence_length": lstm_meta["sequence_length"]}},
        "backtest": _backtest_grid(pred_path, config),
        "artifacts": {"model": _rel(run_dir / "model.pt"), "scaler": _rel(run_dir / "scaler.joblib"),
                      "test_preds": _rel(pred_path),
                      "backtest_curve": _rel(A / "reports" / f"{TICKER}_mlp_backtest_curve.csv"),
                      "perm_importance": _rel(A / "reports" / f"{TICKER}_mlp_perm_importance.csv")},
    }


def _registry_run_entry(run_id: str, config: dict) -> dict:
    """A run trained by DLTrainer on the current processed data and fixed-date split."""
    run_dir = PROJECT_ROOT / "models" / "RELIANCE" / "mlp" / run_id
    meta = json.loads((run_dir / "metadata.json").read_text())
    features = meta["feature_list"]
    X, y = _read_xy(PROJECT_ROOT / config['data'].get('processed_dir', 'data/processed'))
    (Xtr, ytr), (Xv, yv), (Xte, yte) = TimeSeriesSplitter(config).split_train_val_test(X, y)
    selection = json.loads((PROJECT_ROOT / "reports" / f"{TICKER}_feature_selection.json").read_text())
    if selection["selected_features"] != features:
        raise ValueError("Run features differ from the validation-selected feature set")
    raw_dir = PROJECT_ROOT / "data" / "raw"
    raw = pd.read_csv(raw_dir / f"{TICKER}_10y.csv", index_col=0, parse_dates=True)
    nifty = pd.read_csv(raw_dir / "NSEI_10y.csv", index_col=0, parse_dates=True)
    dropped = raw.index.difference(align_trading_calendar(raw, {"NIFTY": nifty}).index)
    sd = config['training']['split_dates']
    return {
        "label": "Phase 3 retrain (cleaned calendar; feature set chosen on validation)",
        "model": _model_block(features, config),
        "data": {"raw_sessions": int(len(raw)), "rows": int(len(X)),
                 "calendar_rule": "NIFTY dates define NSE sessions; zero-volume flat placeholder bars dropped",
                 "dropped_dates": [str(d.date()) for d in dropped],
                 "rows_after_test_end": int((X.index > pd.Timestamp(sd['test_end'])).sum())},
        "split": {"method": f"fixed dates: train <= {sd['train_end']}, val <= {sd['val_end']}, test <= {sd['test_end']}",
                  **_split_summary({"train": (Xtr, ytr), "val": (Xv, yv), "test": (Xte, yte)})},
        "feature_selection": {"split": selection["selection_split"], "metric": selection["selection_metric"],
                              "selected_level": selection["selected_level"],
                              "levels": {lvl: {"n_features": v["n_features"], "run_id": v["run_id"],
                                               **{k: v["val"][k] for k in ("ROC-AUC", "Accuracy", "F1", "Balanced Accuracy")}}
                                         for lvl, v in selection["levels"].items()}},
        "metrics": {"mlp": meta["metrics"],
                    "baselines": _baselines(PROJECT_ROOT / "models", ["majority", "logistic", "random_forest", "xgboost"], features)},
        "backtest": None,
        "pending": ["backtest (Phase 4)", "permutation importance"],
        "artifacts": {"model": _rel(run_dir / "model.pt"), "scaler": _rel(run_dir / "scaler.joblib"),
                      "test_preds": _rel(run_dir / "test_preds.csv")},
    }


def build(new_run_id: str, active: str, path: Path = METRICS_PATH) -> dict:
    config = load_config()
    runs = {OLD_RUN_ID: _old_run_entry(config), new_run_id: _registry_run_entry(new_run_id, config)}
    if active not in runs:
        raise ValueError(f"active run {active} not in {list(runs)}")
    metrics = {"generated_at": datetime.now().isoformat(timespec="seconds"),
               "generated_by": "python main.py metrics", "ticker": TICKER, "target": TARGET,
               "active_run": active, "runs": runs}
    path.write_text(json.dumps(metrics, indent=2))
    logger.info(f"Wrote {path} (active run {active})")
    return metrics
