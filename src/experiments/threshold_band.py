import json
import logging

import pandas as pd

from src.backtesting.engine import Backtester
from src.utils.config import PROJECT_ROOT, load_config

logger = logging.getLogger(__name__)

# Half-widths around 0.5: enter long at P(up) >= 0.5 + h, go to cash at P(up) < 0.5 - h.
# Validation probabilities only span ~0.47-0.55, so the grid is fine-grained near 0.5;
# 0.05 is the requested 0.55 / 0.45 band.
HALF_WIDTHS = [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.05]
EXECUTION = "next_open"
CRITERION = "Excess_Return_vs_BnH"
KEEP = ["Entry_Threshold", "Exit_Threshold", "Days_Tested", "Strategy_Total_Return", "BnH_Total_Return", "Excess_Return_vs_BnH",
        "Strategy_Sharpe", "BnH_Sharpe", "Strategy_Max_Drawdown", "Trades", "Win_Rate_Trade", "Win_Rate_Day",
        "Position_Changes", "Annual_Turnover", "Total_Cost_Paid", "Exposure"]


def result_path(ticker: str):
    return PROJECT_ROOT / "reports" / f"{ticker}_threshold_band.json"


def run_threshold_band(ticker: str, run_id: str) -> dict:
    """Choose a no-trade band on the validation split, then evaluate it on test once."""
    config = load_config()
    run_dir = PROJECT_ROOT / "models" / ticker.replace(".NS", "") / "mlp" / run_id
    val = pd.read_csv(run_dir / "val_preds.csv", index_col=0, parse_dates=True)
    test = pd.read_csv(run_dir / "test_preds.csv", index_col=0, parse_dates=True)
    bt = Backtester(config, PROJECT_ROOT)

    def evaluate(preds, h):
        m = bt.run(ticker, "mlp", threshold=0.5 + h, exit_threshold=0.5 - h, preds=preds, save=False, execution=EXECUTION)
        return {"half_width": h, **{k: m[k] for k in KEEP}}

    val_grid = [evaluate(val, h) for h in HALF_WIDTHS]
    # Best validation excess return over buy & hold; ties go to fewer trades
    chosen = max(val_grid, key=lambda r: (round(r[CRITERION], 10), -r["Trades"]))
    test_result = evaluate(test, chosen["half_width"])
    out = {
        "run_id": run_id, "ticker": ticker, "execution": EXECUTION,
        "selection_split": "validation", "selection_criterion": f"max {CRITERION} (next-open, after costs); ties -> fewer trades",
        "cost_per_side": float(bt.transaction_cost + bt.slippage),
        "validation_prob_range": [float(val["Prob"].min()), float(val["Prob"].max())],
        "validation_grid": val_grid, "chosen_half_width": chosen["half_width"],
        "test_result": test_result,
        "note": "Test evaluated once, for the chosen band only.",
    }
    result_path(ticker).write_text(json.dumps(out, indent=2))
    logger.info(f"Chosen band +/-{chosen['half_width']} on validation; test {test_result['Strategy_Total_Return']:+.2%} "
                f"vs buy & hold {test_result['BnH_Total_Return']:+.2%}")
    return out
