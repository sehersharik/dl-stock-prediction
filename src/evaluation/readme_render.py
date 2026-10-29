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
        pending = f"The backtest for run `{run['run_id']}` has not been produced yet."
        return {"backtest_table": pending, "sensitivity_table": pending, "winrate_note": "", "backtest_rules": pending,
                "header_backtest": "Backtest pending",
                "exec_backtest_row": "| **Backtest** | Pending | Not yet produced for this run |",
                "qa_backtest": "The backtest for the current run has not been produced yet."}
    head_ex = bt["headline_execution"]
    d = bt["variants"][head_ex]
    ret, bnh = _signed_pct(d["Strategy_Total_Return"]), _signed_pct(d["BnH_Total_Return"])
    verdict = "beat" if d["Strategy_Total_Return"] > d["BnH_Total_Return"] else "did **worse** than"
    rules = "\n".join([
        f"- **Out-of-Sample Test Period:** {d['Days_Tested']} signal days ({d['First_Signal_Date']} to {d['Last_Signal_Date']})",
        f"- **Signal:** long if $P(\\text{{UP}}) \\ge {bt['threshold']}$, otherwise cash (cash earns nothing)",
        f"- **Execution (headline):** signal at close $T$; {d['Execution_Rule']}. Also reported: "
        + "; ".join(m["Execution_Rule"] for ex, m in bt["variants"].items() if ex != head_ex),
        f"- **Costs:** {_pct(bt['cost_per_side'])} **per side** (every entry and every exit, including closing at the end); a round trip costs {_pct(2 * bt['cost_per_side'])}",
        f"- **Benchmark:** {bt['benchmark']}",
        f"- **Sharpe:** excess daily returns over a {_pct(bt['risk_free_rate_annual'], 1)} annual risk-free rate, ×√252. {bt['risk_free_source']}",
        f"- **Returns** are measured on the trading-session calendar, so each period is exactly one session. "
        f"{len(bt['sessions_without_prediction'])} sessions in the window have no prediction (input-data gaps) and are skipped by both strategies; "
        f"Reliance's raw price change over the whole window was {_signed_pct(bt['window_price_return'])}.",
    ])
    table = [f"| Metric | Model Strategy ({head_ex.replace('_', '-')}) | Buy & Hold (same periods) |", "|---|---|---|"]
    for label, key, fmt in [("Total Return (after costs)", "Total_Return", _signed_pct), ("Annualized Return", "Annualized", _signed_pct),
                            ("Annualized Volatility", "Volatility", _pct), (f"Sharpe (rf {_pct(bt['risk_free_rate_annual'], 1)})", "Sharpe", lambda x: f"{x:.2f}"),
                            ("Sharpe (rf 0%)", "Sharpe_rf0", lambda x: f"{x:.2f}"), ("Maximum Drawdown", "Max_Drawdown", _pct)]:
        table.append(f"| **{label}** | **{fmt(d['Strategy_' + key])}** | {fmt(d['BnH_' + key])} |")
    table += [
        f"| **Trades (round trips)** | **{d['Trades']}** ({d['Winning_Trades']} winners) | 1 |",
        f"| **Win Rate per Trade** | **{_pct(d['Win_Rate_Trade'], 1)}** (avg trade {_signed_pct(d['Avg_Trade_Return'])}) | N/A |",
        f"| **Win Rate per In-Market Day** | **{_pct(d['Win_Rate_Day'], 1)}** ({d['Winning_Days']} / {d['In_Market_Days']} days) | N/A |",
        f"| **Position Changes / Annual Turnover** | **{d['Position_Changes']}** / {d['Annual_Turnover']:.0f}× capital per year | 2 |",
        f"| **Total Costs Paid** | **{_pct(d['Total_Cost_Paid'])}** of capital | {_pct(2 * bt['cost_per_side'])} |",
        f"| **Market Exposure** | **{_pct(d['Exposure'], 1)}** ({d['In_Market_Days']} long / {d['Cash_Days']} cash days) | 100.0% |",
    ]
    variants = ["| Execution | Strategy Return | Buy & Hold | Sharpe | Max Drawdown | Win Rate / Trade | Win Rate / Day | Trades |",
                "|---|---|---|---|---|---|---|---|"]
    for ex, m in bt["variants"].items():
        variants.append(f"| {m['Execution_Rule']}{' **(headline)**' if ex == head_ex else ''} | {_signed_pct(m['Strategy_Total_Return'])} | "
                        f"{_signed_pct(m['BnH_Total_Return'])} | {m['Strategy_Sharpe']:.2f} | {_pct(m['Strategy_Max_Drawdown'])} | "
                        f"{_pct(m['Win_Rate_Trade'], 1)} | {_pct(m['Win_Rate_Day'], 1)} | {m['Trades']} |")
    backtest_table = "\n".join(table) + "\n\n### Execution Assumption\n\n" + "\n".join(variants)
    if "as_reported" in bt:
        a = bt["as_reported"]
        backtest_table += (f"\n\n> **Previously reported (superseded):** {_signed_pct(a['Strategy_Total_Return'])} vs "
                           f"{_signed_pct(a['BnH_Total_Return'])} buy & hold, Sharpe {a['Strategy_Sharpe']:.2f}. {a['note']}")
    sens = ["| Cost per Side | Transaction Cost + Slippage | Strategy Return | Buy & Hold | Sharpe | Max Drawdown | Win Rate / Trade | Win Rate / Day |",
            "|---|---|---|---|---|---|---|---|"]
    for i, g in enumerate(bt["cost_grid"]):
        label = _pct(g["cost_per_side"]) + (" (default)" if i == bt["default_index"] else "")
        sens.append(f"| **{label}** | {_pct(g['transaction_cost'])} + {_pct(g['slippage'])} | {_signed_pct(g['Strategy_Total_Return'])} | "
                    f"{_signed_pct(g['BnH_Total_Return'])} | {g['Strategy_Sharpe']:.2f} | {_pct(g['Strategy_Max_Drawdown'])} | "
                    f"{_pct(g['Win_Rate_Trade'], 1)} | {_pct(g['Win_Rate_Day'], 1)} |")
    rf_rows = " · ".join(f"rf {float(k):.0%}: {v['Strategy_Sharpe']:.2f} (buy & hold {v['BnH_Sharpe']:.2f})"
                         for k, v in bt["risk_free_sensitivity"].items())
    sens_text = "\n".join(sens) + f"\n\n**Sharpe vs risk-free assumption ({head_ex.replace('_', '-')}, default cost):** {rf_rows}"
    note = ("> **Why per-day win rate and precision can coincide:** with zero costs and *close-to-close* execution, "
            "the strategy is long exactly on the days the model predicts UP and earns exactly the target return, so the "
            f"per-day win rate equals the model's test precision ({_pct(run['metrics']['mlp']['test']['Precision'])}) by construction. "
            "With next-open execution or with costs, the two differ.")
    return {
        "backtest_rules": rules, "backtest_table": backtest_table, "sensitivity_table": sens_text, "winrate_note": note,
        "header_backtest": f"Test backtest ({head_ex.replace('_', '-')}, after costs): strategy {ret} vs {bnh} buy & hold",
        "exec_backtest_row": (f"| **Backtest ({_pct(bt['cost_per_side'])}/side, {head_ex.replace('_', '-')})**| **{ret}** (vs **{bnh}** Buy & Hold, same periods) | "
                              f"Sharpe: **{d['Strategy_Sharpe']:.2f}** \\| Max Drawdown: **{_pct(d['Strategy_Max_Drawdown'])}** \\| "
                              f"Win Rate: **{_pct(d['Win_Rate_Trade'], 1)}** per trade ({d['Trades']} trades) |"),
        "qa_backtest": (f"After {_pct(bt['cost_per_side'])}-per-side costs and executing at the next open, the strategy returned {ret} against "
                        f"{bnh} for buy & hold over the same periods, so it {verdict.replace('**', '')} buy & hold. It traded {d['Trades']} round trips and paid "
                        f"{_pct(d['Total_Cost_Paid'])} of capital in costs. This is one historical period, not evidence of future returns."),
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
