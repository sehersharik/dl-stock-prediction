import logging
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.utils.config import load_config, PROJECT_ROOT
from src.models.dl_trainer import DLTrainer

logger = logging.getLogger(__name__)

# Define logical groupings based on stationary feature generation logic
F_GROUPS = {
    "F0_Base": ["Return_1d", "Return_5d", "Return_10d", "Return_20d", "HL_Spread", "CO_Spread"],
    "Trend": ["Dist_SMA20", "Dist_SMA50", "Dist_SMA200", "SMA20_50_Ratio", "SMA50_200_Ratio"],
    "Momentum": ["RSI_14", "MACD_Ratio", "MACD_Hist_Ratio", "ROC_12"],
    "Volatility": ["ATR_Ratio", "Rolling_Vol_20", "BB_PctB", "BB_Width"],
    "Volume": ["Vol_Change", "Vol_Ratio_20"],
    "Context": ["NIFTY_Return_1d", "NIFTY_Return_5d", "NIFTY_Vol_20", "NIFTY_Dist_SMA50", "VIX_Level", "VIX_Change_1d"]
}

SELECTION_METRIC = "ROC-AUC"


def selection_path(ticker: str) -> Path:
    return PROJECT_ROOT / 'reports' / f"{ticker}_feature_selection.json"


def get_feature_set(X_cols, level=0):
    """Dynamically build the cumulative feature set while ensuring columns actually exist."""
    cols = list(F_GROUPS["F0_Base"])
    if level >= 1: cols += F_GROUPS["Trend"]
    if level >= 2: cols += F_GROUPS["Momentum"]
    if level >= 3: cols += F_GROUPS["Volatility"]
    if level >= 4: cols += F_GROUPS["Volume"]
    if level >= 5: cols += F_GROUPS["Context"]

    # Filter only columns that actually exist in the dataframe
    return [c for c in cols if c in X_cols]


def run_ablation(ticker: str, target_col: str = 'Target_Direction'):
    """Cumulative feature-group ablation for the MLP, scored on the VALIDATION split only.

    The test split is never evaluated here, so the chosen feature set is not influenced
    by test performance. Ties on validation ROC-AUC go to the smaller feature set.
    """
    config = load_config()
    proc_dir = PROJECT_ROOT / config['data'].get('processed_dir', 'data/processed')
    reports_dir = PROJECT_ROOT / 'reports'

    X_full = pd.read_csv(proc_dir / f"{ticker}_X.csv", index_col=0, parse_dates=True)
    all_cols = X_full.columns.tolist()
    trainer = DLTrainer(config, PROJECT_ROOT)

    results = {}
    for level in range(6):
        f_name = f"F{level}"
        feat_cols = get_feature_set(all_cols, level)
        logger.info(f"=== Ablation (validation only): MLP | {f_name} ({len(feat_cols)} features) ===")
        out = trainer.train_mlp(ticker, target_col=target_col, feature_cols=feat_cols, evaluate_test=False)
        results[f_name] = {"run_id": out["run_id"], "n_features": len(feat_cols),
                           "features": feat_cols, "val": out["val"]}
        logger.info(f"{f_name}: val {SELECTION_METRIC} = {out['val'][SELECTION_METRIC]:.4f}")

    best = max(results, key=lambda k: (round(results[k]["val"][SELECTION_METRIC], 6), -results[k]["n_features"]))
    selection = {
        "ticker": ticker,
        "model": "mlp",
        "selection_split": "validation",
        "selection_metric": SELECTION_METRIC,
        "selected_level": best,
        "selected_features": results[best]["features"],
        "levels": results,
    }
    with open(selection_path(ticker), "w") as f:
        json.dump(selection, f, indent=4)
    logger.info(f"Selected {best} ({results[best]['n_features']} features) by validation {SELECTION_METRIC}")

    plot_ablation_results(results, reports_dir, ticker)
    return selection


def plot_ablation_results(results, reports_dir, ticker):
    levels = list(results)
    vals = [results[l]["val"][SELECTION_METRIC] for l in levels]
    plt.figure(figsize=(10, 6))
    plt.plot(levels, vals, marker='o', linewidth=2, label="MLP (validation)")
    plt.axhline(0.5, color='grey', linestyle='--', linewidth=1)
    plt.title(f'Feature Ablation: Validation {SELECTION_METRIC}')
    plt.xlabel('Feature Set (F0=Raw, F5=All)')
    plt.ylabel(f'Validation {SELECTION_METRIC}')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(reports_dir / f"{ticker}_ablation_val_{SELECTION_METRIC}.png")
    plt.close()


if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "RELIANCE.NS"
    run_ablation(ticker)
