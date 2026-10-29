import json
import logging

from src.experiments.feature_ablation import selection_path
from src.models.dl_trainer import DLTrainer
from src.utils.config import PROJECT_ROOT, load_config

logger = logging.getLogger(__name__)

# Fixed in advance (not tuned): two cells x two short lookbacks, project-default architecture
CONFIGS = [("lstm", 10), ("lstm", 20), ("gru", 10), ("gru", 20)]


def result_path(ticker: str):
    return PROJECT_ROOT / "reports" / f"{ticker}_sequence_models.json"


def run_sequence_models(ticker: str) -> dict:
    """Train each sequence model once (early stopping on validation) and score test once."""
    config = load_config()
    features = json.loads(selection_path(ticker).read_text())["selected_features"]
    trainer = DLTrainer(config, PROJECT_ROOT)
    runs = []
    for cell, seq in CONFIGS:
        out = trainer.train_sequence(ticker, cell=cell, seq_length=seq, feature_cols=features)
        runs.append({"name": f"{cell.upper()} ({seq}d)", "cell": cell, "seq_length": seq, "run_id": out["run_id"],
                     "epochs_run": out["epochs_run"], "val": out["val"], "test": out["test"]})
        logger.info(f"{cell.upper()} seq={seq}: val ROC-AUC {out['val']['ROC-AUC']:.4f}, test ROC-AUC {out['test']['ROC-AUC']:.4f}")
    dl_cfg = config["models"].get("deep_learning", {})
    result = {"ticker": ticker, "features": features, "configs_fixed_in_advance": [list(c) for c in CONFIGS],
              "architecture": {"recurrent_dims": dl_cfg.get("lstm_dims", [32, 16]), "dense_dims": dl_cfg.get("dense_dims", [16]),
                               "dropout": dl_cfg.get("dropout", 0.2)},
              "early_stopping": "validation loss, patience 10", "runs": runs}
    result_path(ticker).write_text(json.dumps(result, indent=2))
    return result
