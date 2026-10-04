import pandas as pd
import numpy as np
import ta
import logging

logger = logging.getLogger(__name__)

class FeatureGenerator:
    """
    Calculates technical indicators strictly avoiding lookahead bias.
    All calculations are based on data available at or before time t.
    """
    def __init__(self, config: dict):
        self.config = config
        self.groups = self.config['features']['groups']

    def generate(self, df: pd.DataFrame, context_dfs: dict = None) -> pd.DataFrame:
        """
        Orchestrates feature group generation and handles warm-up NaNs.
        """
        # Ensure we don't modify original dataframe
        df = df.copy()
        initial_rows = len(df)
        
        logger.info("Starting feature generation...")
        
        # Apply toggled groups
        if self.groups.get('raw', False):
            df = self._add_raw(df)
        if self.groups.get('trend', False):
            df = self._add_trend(df)
        if self.groups.get('momentum', False):
            df = self._add_momentum(df)
        if self.groups.get('volatility', False):
            df = self._add_volatility(df)
        if self.groups.get('volume', False):
            df = self._add_volume(df)
        if self.groups.get('market_context', False) and context_dfs is not None:
            df = self._add_market_context(df, context_dfs)
            
        # Dropping warm-up periods explicitly
        # Why: Indicators like SMA_200 require 200 historical points. 
        # The first 199 rows will naturally contain NaNs. We drop them to 
        # provide a perfectly clean feature matrix for models.
        df_clean = df.dropna()
        dropped = initial_rows - len(df_clean)
        
        logger.info(f"Total features generated: {len(df_clean.columns)}")
        logger.info(f"Feature groups enabled: {[k for k, v in self.groups.items() if v]}")
        logger.info(f"Warm-up handling: {dropped} initial rows removed due to required historical lookback.")
        logger.info(f"Missing values remaining in feature matrix: {df_clean.isna().sum().sum()}")
        
        # Guard against duplicate columns
        if df_clean.columns.duplicated().any():
            dup_cols = df_clean.columns[df_clean.columns.duplicated()].tolist()
            raise ValueError(f"Duplicate columns detected during feature generation: {dup_cols}")
            
        return df_clean

    def _add_raw(self, df: pd.DataFrame) -> pd.DataFrame:
        df['Returns'] = df['Close'].pct_change(fill_method=None)
        return df

    def _add_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
        return df

    def _add_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        df['RSI_14'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
        macd = ta.trend.MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9)
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Hist'] = macd.macd_diff()
        df['ROC'] = ta.momentum.ROCIndicator(close=df['Close'], window=12).roc()
        # Simple momentum (price diff)
        df['Momentum'] = df['Close'].diff(periods=10)
        return df

    def _add_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        df['ATR'] = ta.volatility.AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14).average_true_range()
        
        # Rolling Volatility is rolling standard deviation of daily returns
        returns = df['Close'].pct_change(fill_method=None)
        df['Rolling_Vol'] = returns.rolling(window=20).std()
        
        bb = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2)
        df['BB_Upper'] = bb.bollinger_hband()
        df['BB_Middle'] = bb.bollinger_mavg()
        df['BB_Lower'] = bb.bollinger_lband()
        df['BB_Width'] = bb.bollinger_wband()
        return df

    def _add_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        df['Vol_Change'] = df['Volume'].pct_change(fill_method=None)
        df['Vol_SMA_20'] = df['Volume'].rolling(window=20).mean()
        
        # Ratio of current volume to its moving average
        # Using np.where to prevent division by zero in weird edge cases
        df['Vol_Ratio'] = np.where(df['Vol_SMA_20'] > 0, df['Volume'] / df['Vol_SMA_20'], 0)
        
        df['OBV'] = ta.volume.OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume']).on_balance_volume()
        return df

    def _add_market_context(self, df: pd.DataFrame, context_dfs: dict) -> pd.DataFrame:
        """
        Safely merges external index data using an exact timestamp match. 
        Left join strictly prevents lookahead bias from misaligned indices.
        """
        if 'NIFTY' in context_dfs:
            nifty = context_dfs['NIFTY']
            nifty_returns = nifty['Close'].pct_change(fill_method=None)
            nifty_vol = nifty_returns.rolling(20).std()
            
            # Reindex to match the primary dataframe's exact timestamps
            df['NIFTY_Return'] = nifty_returns.reindex(df.index, method=None) # Strictly no forward filling of future data
            df['NIFTY_Vol'] = nifty_vol.reindex(df.index, method=None)
            
        if 'VIX' in context_dfs:
            vix = context_dfs['VIX']
            df['India_VIX'] = vix['Close'].reindex(df.index, method=None)
            
        return df
