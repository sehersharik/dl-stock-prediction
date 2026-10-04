import logging
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.utils.config import load_config, PROJECT_ROOT
from src.models.dl_trainer import DLTrainer

logger = logging.getLogger(__name__)

# Define the logical groupings based on our feature generation logic
F_GROUPS = {
    "F0_Raw": ["Close", "High", "Low", "Open", "Volume", "Returns"],
    "Trend": ["SMA_20", "SMA_50", "SMA_200", "EMA_12", "EMA_26", "EMA_50"],
    "Momentum": ["RSI_14", "MACD", "MACD_Signal", "MACD_Hist", "ROC", "Momentum"],
    "Volatility": ["ATR", "Rolling_Vol", "BB_Upper", "BB_Middle", "BB_Lower", "BB_Width"],
    "Volume": ["Vol_Change", "Vol_SMA_20", "Vol_Ratio", "OBV"],
    "Context": ["NIFTY_Return", "NIFTY_Vol", "India_VIX"] # (If they exist)
}

def get_feature_set(X_cols, level=0):
    """Dynamically build the cumulative feature set while ensuring columns actually exist."""
    cols = list(F_GROUPS["F0_Raw"])
    if level >= 1: cols += F_GROUPS["Trend"]
    if level >= 2: cols += F_GROUPS["Momentum"]
    if level >= 3: cols += F_GROUPS["Volatility"]
    if level >= 4: cols += F_GROUPS["Volume"]
    if level >= 5: cols += F_GROUPS["Context"]
    
    # Filter only columns that actually exist in the dataframe
    return [c for c in cols if c in X_cols]

def run_ablation(ticker: str, target_col: str = 'Target_Direction'):
    config = load_config()
    proc_dir = PROJECT_ROOT / config['data'].get('processed_dir', 'data/processed')
    reports_dir = PROJECT_ROOT / 'reports'
    
    # Read to get column list
    X_full = pd.read_csv(proc_dir / f"{ticker}_X.csv", index_col=0, parse_dates=True)
    all_cols = X_full.columns.tolist()
    
    trainer = DLTrainer(config, PROJECT_ROOT)
    
    results = {}
    models_to_test = ["mlp", "lstm"]
    
    for model_name in models_to_test:
        results[model_name] = {}
        for level in range(6):
            f_name = f"F{level}"
            feat_cols = get_feature_set(all_cols, level)
            logger.info(f"=== Starting Ablation: {model_name.upper()} | {f_name} ({len(feat_cols)} features) ===")
            
            try:
                if model_name == "mlp":
                    metrics = trainer.train_mlp(ticker, target_col=target_col, feature_cols=feat_cols)
                else:
                    metrics = trainer.train_lstm(ticker, target_col=target_col, feature_cols=feat_cols)
                    
                # We only want to save the test metrics dictionary here to avoid nested clutter
                # But trainer returns the full JSON dict which is fine.
                results[model_name][f_name] = metrics
            except Exception as e:
                logger.error(f"Failed {model_name} on {f_name}: {e}")
                results[model_name][f_name] = None

    # Save structured json
    with open(reports_dir / f"{ticker}_ablation_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    # Generate Plots
    plot_ablation_results(results, reports_dir, ticker)
    
def plot_ablation_results(results, reports_dir, ticker):
    metrics_to_plot = ["ROC-AUC", "PR-AUC", "F1", "Accuracy", "Precision", "Recall", "Balanced Accuracy"]
    levels = [f"F{i}" for i in range(6)]
    
    for metric in metrics_to_plot:
        plt.figure(figsize=(10, 6))
        
        for model in ["mlp", "lstm"]:
            y_vals = []
            for lvl in levels:
                res = results[model].get(lvl)
                if res and metric in res:
                    y_vals.append(res[metric])
                else:
                    y_vals.append(np.nan)
                    
            plt.plot(levels, y_vals, marker='o', linewidth=2, label=model.upper())
            
        plt.title(f'Feature Ablation Impact on {metric}')
        plt.xlabel('Feature Set (F0=Raw, F5=All)')
        plt.ylabel(metric)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(reports_dir / f"{ticker}_ablation_{metric}.png")
        plt.close()
        logger.info(f"Saved ablation plot for {metric}.")

if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "RELIANCE.NS"
    run_ablation(ticker)
