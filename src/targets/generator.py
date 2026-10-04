import pandas as pd
import logging

logger = logging.getLogger(__name__)

class TargetGenerator:
    """
    Generates classification targets for next-day direction and downside risk.
    Strictly isolates targets from features to prevent data leakage.
    """
    def __init__(self, config: dict):
        self.config = config
        self.risk_threshold = self.config['targets'].get('downside_risk_threshold', -0.02)
        
    def generate(self, df: pd.DataFrame, drop_na_targets: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Takes the feature matrix, calculates the t+1 targets, and returns cleanly isolated X and y dataframes.
        """
        df = df.copy()
        
        if 'Close' not in df.columns:
            raise ValueError("Feature matrix must contain 'Close' price to compute targets.")
            
        # Target Calculation
        # shift(-1) moves tomorrow's close onto today's row to compute the future return
        next_day_return = df['Close'].shift(-1) / df['Close'] - 1
        
        y = pd.DataFrame(index=df.index)
        y['Next_Day_Return'] = next_day_return
        y['Target_Direction'] = (next_day_return > 0).astype(int)
        y['Target_Risk'] = (next_day_return <= self.risk_threshold).astype(int)
        
        # The very last row lacks a tomorrow, resulting in NaN return. 
        # For historical training/eval, we must drop it.
        if drop_na_targets:
            valid_idx = y['Next_Day_Return'].notna()
            df = df.loc[valid_idx]
            y = y.loc[valid_idx]
            
        X = df.copy()
        
        # Validation 1: Leakage check
        forbidden_cols = ['Next_Day_Return', 'Target_Direction', 'Target_Risk']
        leakage = [col for col in forbidden_cols if col in X.columns]
        if leakage:
            raise ValueError(f"CRITICAL: Target leakage detected in feature matrix X! Found: {leakage}")
            
        # Validation 2: Index Alignment
        if not X.index.equals(y.index):
            raise ValueError("CRITICAL: Index misalignment between features (X) and targets (y).")
            
        self._log_statistics(y)
        
        return X, y
        
    def _log_statistics(self, y: pd.DataFrame):
        total = len(y)
        if total == 0:
            logger.warning("Target DataFrame is empty. No stats to report.")
            return
            
        dir_pos = y['Target_Direction'].sum()
        risk_pos = y['Target_Risk'].sum()
        
        stats = {
            "Total Valid Rows": total,
            "Direction (Up) Count": int(dir_pos),
            "Direction (Up) %": round((dir_pos / total) * 100, 2),
            "Risk (Down) Count": int(risk_pos),
            "Risk (Down) %": round((risk_pos / total) * 100, 2),
            "Min Future Return": round(y['Next_Day_Return'].min(), 4),
            "Max Future Return": round(y['Next_Day_Return'].max(), 4)
        }
        
        logger.info("=== Target Generation Statistics ===")
        for k, v in stats.items():
            logger.info(f"  {k}: {v}")
