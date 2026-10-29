"""Render README.md from docs/README.template.md and the active run in reports/metrics.json.

Every result number in the README is a `{{placeholder}}` filled here, so the README can
only show values that `python main.py metrics` produced. Run `python main.py readme`
after rebuilding metrics.json; tests/test_metrics_registry.py fails if they drift apart.
"""
import re
from pathlib import Path

import pandas as pd

from src.evaluation.metrics_registry import active_run, load_metrics
from src.utils.config import PROJECT_ROOT

TEMPLATE_PATH = PROJECT_ROOT / "docs" / "README.template.md"
README_PATH = PROJECT_ROOT / "README.md"
PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")

LEVEL_DESCRIPTIONS = {
    "F0": ("F0 (Base)", "Returns + Spreads"),
    "F1": ("F1 (Trend)", "F0 + Moving Average Ratios"),
    "F2": ("F2 (Momentum)", "F1 + RSI, MACD, ROC"),
    "F3": ("F3 (Volatility)", "F2 + ATR, Volatility, Bollinger"),
    "F4": ("F4 (Volume)", "F3 + Volume Changes & Ratios"),
    "F5": ("F5 (Context)", "F4 + NIFTY 50 & India VIX"),
}
BASELINE_NAMES = {"majority": "Majority Class Baseline", "logistic": "Logistic Regression",
                  "random_forest": "Random Forest", "xgboost": "XGBoost"}


def _pct(x, d=2):
    return f"{x * 100:.{d}f}%"


def _signed_pct(x):
    return f"{x * 100:+.2f}%"


def _metric_row(name, m, bold=False):
    cells = [_pct(m["Accuracy"]), _pct(m["Balanced Accuracy"]), _pct(m["Precision"]), _pct(m["Recall"]),
             f"{m['F1']:.4f}", f"{m['ROC-AUC']:.4f}", f"{m['PR-AUC']:.4f}"]
    if bold:
        name, cells = f"**{name}**", [f"**{c}**" for c in cells]
    return f"| {name} | " + " | ".join(cells) + " |"


def _results_table(run):
    mets = run["metrics"]
    rows = ["| Model Architecture | Accuracy | Balanced Acc | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |",
            "|---|---|---|---|---|---|---|---|"]
    for key, name in BASELINE_NAMES.items():
        if key in mets.get("baselines", {}):
            rows.append(_metric_row(name, mets["baselines"][key]["test"]))
    layers = run["model"]["layers"]
    rows.append(_metric_row(f"Small MLP {layers} (Final)", mets["mlp"]["test"], bold=True))
    if "lstm" in mets:
        rows.append(_metric_row(f"LSTM (Seq {mets['lstm']['sequence_length']})", mets["lstm"]["test"]))
    note = (f"> All rows use the same split, feature set ({run['model']['n_features']} features) and threshold "
            f"({run['model']['threshold']}). Source: `reports/metrics.json`, run `{run['run_id']}`.")
    return "\n".join(rows) + "\n\n" + note


def _ablation_table(run):
    sel = run["feature_selection"]
    split = sel["split"].capitalize()
    lstm = sel.get("lstm_levels", {})
    head = f"| Feature Level | Feature Groups Included | Total Features | MLP {split} ROC-AUC | MLP {split} Accuracy | MLP {split} F1 |"
    sep = "|---|---|---|---|---|---|"
    if lstm:
        head += f" LSTM {split} ROC-AUC |"
        sep += "---|"
    rows = [head, sep]
    for lvl, v in sel["levels"].items():
        label, desc = LEVEL_DESCRIPTIONS[lvl]
        n = v.get("n_features", run["model"]["n_features"] if lvl == "F5" else "")
        cells = [str(n), f"{v['ROC-AUC']:.4f}", _pct(v["Accuracy"]), f"{v['F1']:.4f}"]
        if lstm:
            cells.append(f"{lstm[lvl]['ROC-AUC']:.4f}" if lvl in lstm else "—")
        if lvl == sel["selected_level"]:
            label, cells = f"{label} ✔ selected", [f"**{c}**" for c in cells]
        rows.append(f"| **{label}** | {desc} | " + " | ".join(cells) + " |")
    if sel["split"] == "test":
        caveat = ("> **Caveat — test-set contamination:** these ablation scores were measured on the test period, and the "
                  f"final feature set ({sel['selected_level']}) was chosen because it scored best there. The headline test "
                  f"ROC-AUC of {run['metrics']['mlp']['test']['ROC-AUC']:.4f} is therefore likely optimistic.")
    else:
        caveat = (f"> Feature set chosen by **{sel['split']} {sel['metric']} only**; the test period was not scored during "
                  "ablation and was evaluated once, after the choice was frozen.")
    return "\n".join(rows) + "\n\n" + caveat


def _perm_block(run):
    path = run["artifacts"].get("perm_importance")
    if not path or not (PROJECT_ROOT / path).exists():
        return f"Permutation importance has not been computed for run `{run['run_id']}` yet."
    df = pd.read_csv(PROJECT_ROOT / path)
    feat_col, imp_col = df.columns[0], df.columns[1]
    lines = ["```text", "Top Features Ranked by Drop in Test ROC-AUC:", ""]
    for i, (_, r) in enumerate(df.head(10).iterrows(), 1):
        lines.append(f"{i}. {r[feat_col]:<18} [{r[imp_col]:+.4f}]")
    lines.append("```")
    return "\n".join(lines)


def _backtest_blocks(run):
    bt = run.get("backtest")
    if not bt:
        pending = f"The backtest for run `{run['run_id']}` has not been produced yet (Phase 4)."
        return {"backtest_table": pending, "sensitivity_table": pending, "winrate_note": "",
                "header_backtest": "Backtest pending (Phase 4)",
                "exec_backtest_row": "| **Backtest** | Pending | Not yet produced for this run (Phase 4) |",
                "default_cost": "n/a", "default_cost_parts": "n/a",
                "qa_backtest": "The backtest for the current run has not been produced yet (Phase 4)."}
    d = bt["cost_grid"][bt["default_index"]]
    exposure = d["In_Market_Days"] / d["Days_Tested"]
    table = "\n".join([
        f"| Metric | Model Strategy (MLP {run['model']['layers']}) | Buy & Hold Benchmark |",
        "|---|---|---|",
        f"| **Cumulative Total Return** | **{_signed_pct(d['Strategy_Total_Return'])}** | {_signed_pct(d['BnH_Total_Return'])} |",
        f"| **Annualized Return** | **{_signed_pct(d['Strategy_Annualized'])}** | {_signed_pct(d['BnH_Annualized'])} |",
        f"| **Annualized Volatility** | **{_pct(d['Strategy_Volatility'])}** | {_pct(d['BnH_Volatility'])} |",
        f"| **Sharpe Ratio** | **{d['Strategy_Sharpe']:.2f}** | {d['BnH_Sharpe']:.2f} |",
        f"| **Maximum Drawdown** | **{_pct(d['Strategy_Max_Drawdown'])}** | {_pct(d['BnH_Max_Drawdown'])} |",
        f"| **Win Rate (per in-market day)** | **{_pct(d['Win_Rate'])}** ({d['Winning_Days']} wins / {d['In_Market_Days']} in-market days) | N/A |",
        f"| **Position Changes** | **{d['Total_Trades']}** | 1 (Buy at start) |",
        f"| **Market Exposure** | **{_pct(exposure, 1)}** ({d['In_Market_Days']} long days, {d['Cash_Days']} cash days) | 100.0% |",
    ])
    sens = ["| Cost per Position Change | Transaction Cost + Slippage | Cumulative Return | Sharpe Ratio | Max Drawdown | Win Rate (per day) |",
            "|---|---|---|---|---|---|"]
    for i, g in enumerate(bt["cost_grid"]):
        label = f"{_pct(g['cost_per_position_change'])}" + (" (default)" if i == bt["default_index"] else "")
        sens.append(f"| **{label}** | {_pct(g['transaction_cost'])} + {_pct(g['slippage'])} | {_signed_pct(g['Strategy_Total_Return'])} | "
                    f"{g['Strategy_Sharpe']:.2f} | {_pct(g['Strategy_Max_Drawdown'])} | {_pct(g['Win_Rate'])} |")
    zero = next((g for g in bt["cost_grid"] if g["cost_per_position_change"] == 0), None)
    note = ""
    if zero is not None:
        note = (f"> **Why the zero-cost win rate equals the MLP's precision ({_pct(zero['Win_Rate'])}):** win rate here is counted per "
                f"in-market day. At threshold {bt['threshold']} the strategy is in the market exactly on the days the model "
                "predicts UP, so with no costs a winning day is the same as a correct UP prediction.")
    ret, bnh = _signed_pct(d["Strategy_Total_Return"]), _signed_pct(d["BnH_Total_Return"])
    return {
        "backtest_table": table, "sensitivity_table": "\n".join(sens), "winrate_note": note,
        "header_backtest": f"Out-of-Sample Test Backtest {ret} vs {bnh} Buy & Hold (Sharpe {d['Strategy_Sharpe']:.2f})",
        "exec_backtest_row": (f"| **Backtest Return ({_pct(d['cost_per_position_change'])} Cost)**| **{ret}** (vs **{bnh}** Buy & Hold) | "
                              f"Sharpe: **{d['Strategy_Sharpe']:.2f}** \\| Max Drawdown: **{_pct(d['Strategy_Max_Drawdown'])}** \\| "
                              f"Win Rate: **{_pct(d['Win_Rate'])}** ({d['Total_Trades']} position changes) |"),
        "default_cost": _pct(d["cost_per_position_change"]),
        "default_cost_parts": f"{_pct(d['transaction_cost'])} transaction cost + {_pct(d['slippage'])} slippage",
        "qa_backtest": (f"Over the test period the strategy returned {ret} against {bnh} for Buy & Hold. The model spent "
                        f"{_pct(d['Cash_Days'] / d['Days_Tested'], 1)} of trading days in cash, so most of the difference comes "
                        "from avoiding drawdowns rather than from frequent correct calls. This is one historical period, not evidence of future returns."),
    }


def context(run: dict) -> dict:
    split, mlp, sel = run["split"], run["metrics"]["mlp"]["test"], run["feature_selection"]
    layers = run["model"]["layers"]
    dims = [run["model"]["n_features"], *layers, 1]
    preds = pd.read_csv(PROJECT_ROOT / run["artifacts"]["test_preds"])["Prob"]
    levels = list(sel["levels"].values())
    ctx = {
        "run_id": run["run_id"],
        "raw_sessions": f"{run['data']['raw_sessions']:,}", "rows": f"{run['data']['rows']:,}",
        "calendar_rule": run["data"]["calendar_rule"], "split_method": split["method"],
        "n_params": f"{sum(a * b + b for a, b in zip(dims[:-1], dims[1:])):,}",
        "prob_min": f"{preds.min():.3f}", "prob_max": f"{preds.max():.3f}", "prob_std": f"{preds.std():.3f}",
        "test_years": f"{split['test']['start'][:4]}–{split['test']['end'][:4]}",
        "test_roc": f"{mlp['ROC-AUC']:.4f}", "test_balacc": _pct(mlp["Balanced Accuracy"]),
        "test_precision": _pct(mlp["Precision"]), "test_recall": _pct(mlp["Recall"]), "test_f1": f"{mlp['F1']:.4f}",
        "selection_caveat_short": ("also used to choose the feature set, so likely optimistic" if sel["split"] == "test"
                                   else f"evaluated once; feature set chosen on {sel['split']}"),
        "selection_split": sel["split"],
        "abl_prev": f"{levels[-2]['ROC-AUC']:.3f}", "abl_last": f"{levels[-1]['ROC-AUC']:.3f}",
        "results_table": _results_table(run), "ablation_table": _ablation_table(run), "perm_block": _perm_block(run),
        **_backtest_blocks(run),
    }
    for name in ("train", "val", "test"):
        s = split[name]
        ctx.update({f"n_{name}": f"{s['n']:,}", f"{name}_start": s["start"], f"{name}_end": s["end"],
                    f"up_{name}": _pct(s["up_share"], 1)})
    return ctx


def render_text(template: str, run: dict) -> str:
    ctx = context(run)
    missing = sorted(set(PLACEHOLDER.findall(template)) - set(ctx))
    if missing:
        raise KeyError(f"README template uses unknown placeholders: {missing}")
    return PLACEHOLDER.sub(lambda m: ctx[m.group(1)], template)


def render(template_path: Path = TEMPLATE_PATH, out_path: Path = README_PATH) -> str:
    text = render_text(template_path.read_text(), active_run(load_metrics()))
    out_path.write_text(text)
    return text
