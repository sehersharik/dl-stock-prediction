import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging
import matplotlib.pyplot as plt
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class TimeSeriesSplitter:
    """
    Handles chronological train/val/test splits and expanding-window 
    walk-forward validation to strictly prevent look-ahead bias.
    """
    def __init__(self, config: dict):
        self.config = config
        # Use default config values or fallbacks
        training_config = self.config.get('training', {})
        self.train_pct = training_config.get('train_split', 0.70)
        self.val_pct = training_config.get('val_split', 0.15)
        
    def split_train_val_test(self, X: pd.DataFrame, y: pd.DataFrame) -> tuple:
        """
        Chronologically splits data into Train, Val, Test.
        Returns: (X_train, y_train), (X_val, y_val), (X_test, y_test)
        """
        n = len(X)
        train_end = int(n * self.train_pct)
        val_end = int(n * (self.train_pct + self.val_pct))
        
        X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
        X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
        X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]
        
        # Enforce strict chronology
        self.validate_chronology(X_train, X_val, X_test)
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
        
    def walk_forward_splits(self, X: pd.DataFrame, y: pd.DataFrame, n_splits: int = 5, initial_train_pct: float = 0.5):
        """
        Generates expanding-window walk-forward splits.
        Yields (X_train, y_train, X_val, y_val) tuples for each fold.
        """
        n = len(X)
        initial_train_end = int(n * initial_train_pct)
        remaining = n - initial_train_end
        val_size = remaining // n_splits
        
        splits = []
        for i in range(n_splits):
            train_end = initial_train_end + (i * val_size)
            val_end = train_end + val_size
            
            # For the last fold, take the remainder
            if i == n_splits - 1:
                val_end = n
                
            X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
            X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
            
            self.validate_chronology(X_train, X_val)
            splits.append((X_train, y_train, X_val, y_val))
            
        return splits

    def scale_features(self, X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame = None):
        """
        Strictly fits the scaler on X_train ONLY, then transforms Val and Test.
        """
        scaler = StandardScaler()
        
        # FIT ONLY ON TRAIN
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
        
        # TRANSFORM ONLY ON VAL AND TEST
        X_val_scaled = pd.DataFrame(scaler.transform(X_val), index=X_val.index, columns=X_val.columns)
        
        if X_test is not None:
            X_test_scaled = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)
            return scaler, X_train_scaled, X_val_scaled, X_test_scaled
            
        return scaler, X_train_scaled, X_val_scaled

    @staticmethod
    def validate_chronology(train, val, test=None):
        """Asserts mathematically that no data overlap exists and order is strictly temporal."""
        if len(train) == 0 or len(val) == 0:
            return
            
        # Assertion 1: Train comes before Validation
        assert train.index.max() < val.index.min(), f"Leakage: Train end {train.index.max()} >= Val start {val.index.min()}"
        
        if test is not None and len(test) > 0:
            # Assertion 2: Validation comes before Test
            assert val.index.max() < test.index.min(), f"Leakage: Val end {val.index.max()} >= Test start {test.index.min()}"

    def plot_splits(self, X: pd.DataFrame, save_path: Path):
        """Generates a visual Gantt-style chart of the training timeline."""
        (X_train, _), (X_val, _), (X_test, _) = self.split_train_val_test(X, X) # Dummy y
        
        plt.figure(figsize=(10, 3))
        plt.barh("Train", (X_train.index.max() - X_train.index.min()).days, left=X_train.index.min().toordinal(), color="blue")
        plt.barh("Validation", (X_val.index.max() - X_val.index.min()).days, left=X_val.index.min().toordinal(), color="orange")
        plt.barh("Test", (X_test.index.max() - X_test.index.min()).days, left=X_test.index.min().toordinal(), color="green")
        
        # Format axes
        import matplotlib.dates as mdates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        
        plt.title("Chronological Train / Val / Test Split")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Split visualization saved to {save_path}")

    def log_walk_forward_table(self, splits: list):
        """Outputs an ASCII table mapping the walk forward folds."""
        logger.info("=== Walk-Forward Validation Folds ===")
        header = f"{'Fold':<6} | {'Train Start':<12} | {'Train End':<12} | {'Val Start':<12} | {'Val End':<12}"
        logger.info(header)
        logger.info("-" * len(header))
        for i, (X_tr, _, X_v, _) in enumerate(splits):
            tr_start = str(X_tr.index.min().date())
            tr_end = str(X_tr.index.max().date())
            v_start = str(X_v.index.min().date())
            v_end = str(X_v.index.max().date())
            logger.info(f"Fold {i+1:<1} | {tr_start:<12} | {tr_end:<12} | {v_start:<12} | {v_end:<12}")
