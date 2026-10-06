import pandas as pd
import logging

logger = logging.getLogger(__name__)

class TargetGenerator:
    """
    Generates classification targets for next-day direction and downside risk.
    Strictly isolates targets from features to prevent data leakage.
    Accepts raw price series to compute future returns without needing 'Close' in X.
    """
    def __init__(self, config: dict):
        self.config = config
        self.risk_threshold = self.config['targets'].get('downside_risk_threshold', -0.02)
        
    def generate(self, df_features: pd.DataFrame, df_prices: pd.DataFrame, drop_na_targets: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Takes the feature matrix and price history, calculates t+1 targets, 
        and returns cleanly isolated and aligned X and y dataframes.
        """
        if 'Close' not in df_prices.columns:
            raise ValueError("Price dataframe must contain 'Close' price to compute future targets.")
            
        close = df_prices['Close'].reindex(df_features.index)
        
        # Target Calculation
        # shift(-1) moves tomorrow's close onto today's row to compute the future return
        next_day_return = close.shift(-1) / close - 1
        
        y = pd.DataFrame(index=df_features.index)
        y['Next_Day_Return'] = next_day_return
        y['Target_Direction'] = (next_day_return > 0).astype(int)
        y['Target_Risk'] = (next_day_return <= self.risk_threshold).astype(int)
        
        X = df_features.copy()
        
        # The very last row lacks a tomorrow, resulting in NaN return.
        if drop_na_targets:
            valid_idx = y['Next_Day_Return'].notna() & X.notna().all(axis=1)
            X = X.loc[valid_idx]
            y = y.loc[valid_idx]
            
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
