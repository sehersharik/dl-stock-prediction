import pandas as pd
import numpy as np
import joblib
import json
import logging
from pathlib import Path

from src.evaluation.splitter import TimeSeriesSplitter
from src.evaluation.metrics import evaluate_classification
from src.models.baselines import get_baseline_model

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Orchestrates chronologically sound training, prediction, and artifact serialization."""
    def __init__(self, config: dict, project_root: Path):
        self.config = config
        self.project_root = project_root
        self.splitter = TimeSeriesSplitter(config)
        
    def train_baseline(self, ticker: str, model_name: str, target_col: str = 'Target_Direction'):
        proc_dir = self.project_root / self.config['data'].get('processed_dir', 'data/processed')
        models_dir = self.project_root / 'models'
        pred_dir = self.project_root / 'data' / 'predictions'
        models_dir.mkdir(parents=True, exist_ok=True)
        pred_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Load Isolated Data
        X = pd.read_csv(proc_dir / f"{ticker}_X.csv", index_col=0, parse_dates=True)
        y = pd.read_csv(proc_dir / f"{ticker}_y.csv", index_col=0, parse_dates=True)
        y_target = y[target_col]
        
        # 2. Chronological Split
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = self.splitter.split_train_val_test(X, y_target)
        
        # 3. Strict Scaling
        scaler, X_train_s, X_val_s, X_test_s = self.splitter.scale_features(X_train, X_val, X_test)
        
        # 4. Initialize & Fit
        logger.info(f"Training {model_name} on {ticker} for {target_col}...")
        model = get_baseline_model(model_name)
        model.fit(X_train_s, y_train)
        
        # 5. Predict & Evaluate
        def evaluate_set(X_set, y_set, split_name):
            y_pred = model.predict(X_set)
            y_prob = model.predict_proba(X_set)[:, 1] if hasattr(model, "predict_proba") else None
            metrics = evaluate_classification(y_set, y_pred, y_prob)
            
            # Save predictions
            df_pred = pd.DataFrame({'True': y_set, 'Pred': y_pred, 'Prob': y_prob}, index=y_set.index)
            df_pred.to_csv(pred_dir / f"{ticker}_{model_name}_{target_col}_{split_name}_preds.csv")
            return metrics
            
        train_metrics = evaluate_set(X_train_s, y_train, "train")
        val_metrics = evaluate_set(X_val_s, y_val, "val")
        test_metrics = evaluate_set(X_test_s, y_test, "test")
        
        # 6. Save Artifacts
        artifacts = {
            "model_name": model_name,
            "ticker": ticker,
            "target": target_col,
            "metrics": {
                "train": train_metrics,
                "val": val_metrics,
                "test": test_metrics
            }
        }
        
        with open(models_dir / f"{ticker}_{model_name}_{target_col}_metrics.json", "w") as f:
            json.dump(artifacts, f, indent=4)
            
        joblib.dump(model, models_dir / f"{ticker}_{model_name}_{target_col}.joblib")
        joblib.dump(scaler, models_dir / f"{ticker}_{model_name}_{target_col}_scaler.joblib")
        
        logger.info(f"Finished {model_name}. Validation ROC-AUC: {val_metrics.get('ROC-AUC')}")
        return artifacts
