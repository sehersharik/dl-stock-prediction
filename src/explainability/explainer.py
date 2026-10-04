import numpy as np
import pandas as pd
import torch
import json
import logging
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_auc_score
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

logger = logging.getLogger(__name__)

F_GROUPS = {
    "Raw": ["Close", "High", "Low", "Open", "Volume", "Returns"],
    "Trend": ["SMA_20", "SMA_50", "SMA_200", "EMA_12", "EMA_26", "EMA_50"],
    "Momentum": ["RSI_14", "MACD", "MACD_Signal", "MACD_Hist", "ROC", "Momentum"],
    "Volatility": ["ATR", "Rolling_Vol", "BB_Upper", "BB_Middle", "BB_Lower", "BB_Width"],
    "Volume": ["Vol_Change", "Vol_SMA_20", "Vol_Ratio", "OBV"],
    "Context": ["NIFTY_Return", "NIFTY_Vol", "India_VIX"]
}

class ModelExplainer:
    def __init__(self, model, device, feature_names: list, reports_dir: Path, ticker: str, model_name: str):
        self.model = model
        self.device = device
        self.feature_names = feature_names
        self.reports_dir = reports_dir
        self.ticker = ticker
        self.model_name = model_name
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
    def predict_prob(self, X_np):
        self.model.eval()
        with torch.no_grad():
            t = torch.FloatTensor(X_np).to(self.device)
            logits = self.model(t)
            return torch.sigmoid(logits).cpu().numpy().flatten()
            
    def run_permutation_importance(self, X: np.ndarray, y: np.ndarray, n_repeats: int = 5):
        """Calculates permutation importance using ROC-AUC as the baseline metric."""
        logger.info("Running permutation importance...")
        baseline_preds = self.predict_prob(X)
        try:
            baseline_score = roc_auc_score(y, baseline_preds)
        except ValueError:
            logger.warning("ROC-AUC failed (likely single class in y). Falling back to accuracy.")
            from sklearn.metrics import accuracy_score
            baseline_score = accuracy_score(y, baseline_preds >= 0.5)
            
        importances = np.zeros((X.shape[1], n_repeats))
        
        for i in range(X.shape[1]):
            for r in range(n_repeats):
                X_shuffled = X.copy()
                np.random.shuffle(X_shuffled[:, i])
                shuff_preds = self.predict_prob(X_shuffled)
                try:
                    score = roc_auc_score(y, shuff_preds)
                except:
                    score = accuracy_score(y, shuff_preds >= 0.5)
                # Importance is drop in performance
                importances[i, r] = baseline_score - score
                
        means = np.mean(importances, axis=1)
        stds = np.std(importances, axis=1)
        
        # Save structured results
        imp_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance_Mean': means,
            'Importance_Std': stds
        }).sort_values('Importance_Mean', ascending=False)
        
        imp_df.to_csv(self.reports_dir / f"{self.ticker}_{self.model_name}_perm_importance.csv", index=False)
        self._plot_permutation_importance(imp_df)
        self._plot_group_importance(imp_df, "Permutation")
        
        return imp_df
        
    def _plot_permutation_importance(self, imp_df: pd.DataFrame, top_n: int = 15):
        plt.figure(figsize=(10, 8))
        plot_df = imp_df.head(top_n).sort_values('Importance_Mean', ascending=True)
        plt.barh(plot_df['Feature'], plot_df['Importance_Mean'], xerr=plot_df['Importance_Std'], color='skyblue')
        plt.title(f'Top {top_n} Features by Permutation Importance\n(Note: Importance to MODEL, not causal to market)')
        plt.xlabel('Drop in ROC-AUC')
        plt.tight_layout()
        plt.savefig(self.reports_dir / f"{self.ticker}_{self.model_name}_perm_importance.png")
        plt.close()
        
    def _plot_group_importance(self, imp_df: pd.DataFrame, method: str):
        group_scores = {k: 0.0 for k in F_GROUPS.keys()}
        
        for _, row in imp_df.iterrows():
            feat = row['Feature']
            score = row['Importance_Mean'] if method == "Permutation" else row['Mean_Abs_SHAP']
            # Find group
            for g_name, g_feats in F_GROUPS.items():
                if feat in g_feats:
                    group_scores[g_name] += score
                    break
                    
        # Filter groups that had features
        group_scores = {k: v for k, v in group_scores.items() if v != 0}
        
        plt.figure(figsize=(8, 5))
        groups = list(group_scores.keys())
        scores = list(group_scores.values())
        
        # Sort
        sorted_indices = np.argsort(scores)
        groups = [groups[i] for i in sorted_indices]
        scores = [scores[i] for i in sorted_indices]
        
        plt.barh(groups, scores, color='coral')
        plt.title(f'Feature Group Importance ({method})\n(Contribution to Model Predictions, Not Causal)')
        plt.xlabel(f'Aggregated {method} Score')
        plt.tight_layout()
        plt.savefig(self.reports_dir / f"{self.ticker}_{self.model_name}_{method}_group_importance.png")
        plt.close()
        
        with open(self.reports_dir / f"{self.ticker}_{self.model_name}_{method}_groups.json", "w") as f:
            json.dump(group_scores, f, indent=4)

    def run_shap(self, X_train: np.ndarray, X_test: np.ndarray):
        if not SHAP_AVAILABLE:
            logger.warning("SHAP is not installed. Skipping SHAP analysis.")
            return
            
        logger.info("Running SHAP DeepExplainer analysis...")
        self.model.eval()
        
        # Use a background subset to save memory
        bg_size = min(100, len(X_train))
        background = torch.FloatTensor(X_train[np.random.choice(X_train.shape[0], bg_size, replace=False)]).to(self.device)
        test_tensor = torch.FloatTensor(X_test).to(self.device)
        
        try:
            explainer = shap.DeepExplainer(self.model, background)
            shap_values = explainer.shap_values(test_tensor)
            
            # DeepExplainer returns a list of arrays for classification, we want the magnitude
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
                
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_test, feature_names=self.feature_names, show=False)
            plt.title("SHAP Summary Plot (Model Contribution, Not Causal)")
            plt.tight_layout()
            plt.savefig(self.reports_dir / f"{self.ticker}_{self.model_name}_shap_summary.png")
            plt.close()
            
            # Feature group importances for SHAP
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            shap_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Mean_Abs_SHAP': mean_abs_shap
            })
            self._plot_group_importance(shap_df, "SHAP")
            
        except Exception as e:
            logger.error(f"SHAP DeepExplainer failed (sometimes incompatible with certain PyTorch ops): {e}")

