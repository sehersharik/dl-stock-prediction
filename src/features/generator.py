import pandas as pd
import numpy as np
import ta
import logging

logger = logging.getLogger(__name__)

class FeatureGenerator:
    """
    Calculates technical indicators strictly avoiding lookahead bias.
    All calculations are based on data available at or before time t.
    All features are stationary, scale-invariant, and dimensionless.
    """
    def __init__(self, config: dict):
        self.config = config
        self.groups = self.config['features'].get('groups', {})

    def generate(self, df: pd.DataFrame, context_dfs: dict = None) -> pd.DataFrame:
        df = df.copy()
        initial_rows = len(df)
        
        logger.info("Starting stationary feature generation...")
        
        close = df['Close']
        high = df['High']
        low = df['Low']
        open_p = df['Open']
        vol = df['Volume'].replace(0, np.nan).ffill()
        
        feats = pd.DataFrame(index=df.index)
        
        # 1. Stationary Returns & Intraday Spreads (Raw / Base)
        if self.groups.get('raw', True):
            feats['Return_1d'] = close.pct_change()
            feats['Return_5d'] = close.pct_change(5)
            feats['Return_10d'] = close.pct_change(10)
            feats['Return_20d'] = close.pct_change(20)
            feats['HL_Spread'] = (high - low) / close
            feats['CO_Spread'] = (close - open_p) / open_p
        
        # 2. Stationary Trend Ratios (Distance to Moving Averages)
        if self.groups.get('trend', True):
            sma20 = close.rolling(20).mean()
            sma50 = close.rolling(50).mean()
            sma200 = close.rolling(200).mean()
            feats['Dist_SMA20'] = (close / sma20) - 1
            feats['Dist_SMA50'] = (close / sma50) - 1
            feats['Dist_SMA200'] = (close / sma200) - 1
            feats['SMA20_50_Ratio'] = (sma20 / sma50) - 1
            feats['SMA50_200_Ratio'] = (sma50 / sma200) - 1
        
        # 3. Normalized Momentum Oscillators
        if self.groups.get('momentum', True):
            feats['RSI_14'] = ta.momentum.RSIIndicator(close=close, window=14).rsi() / 100.0
            macd = ta.trend.MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
            feats['MACD_Ratio'] = macd.macd() / close
            feats['MACD_Hist_Ratio'] = macd.macd_diff() / close
            feats['ROC_12'] = ta.momentum.ROCIndicator(close=close, window=12).roc() / 100.0
        
        # 4. Normalized Volatility Metrics & Bands
        if self.groups.get('volatility', True):
            atr = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()
            feats['ATR_Ratio'] = atr / close
            feats['Rolling_Vol_20'] = close.pct_change().rolling(20).std()
            bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
            bb_w = bb.bollinger_hband() - bb.bollinger_lband()
            feats['BB_PctB'] = np.where(bb_w > 0, (close - bb.bollinger_lband()) / bb_w, 0.5)
            feats['BB_Width'] = np.where(bb.bollinger_mavg() > 0, bb_w / bb.bollinger_mavg(), 0.0)
        
        # 5. Relative Volume Metrics
        if self.groups.get('volume', True):
            vol_sma20 = vol.rolling(20).mean()
            feats['Vol_Change'] = vol.pct_change()
            feats['Vol_Ratio_20'] = np.where(vol_sma20 > 0, vol / vol_sma20, 1.0)
        
        # 6. Market Context (NIFTY 50 + India VIX)
        if self.groups.get('market_context', True) and context_dfs:
            if 'NIFTY' in context_dfs and context_dfs['NIFTY'] is not None and not context_dfs['NIFTY'].empty:
                n_close = context_dfs['NIFTY']['Close'].reindex(df.index)
                feats['NIFTY_Return_1d'] = n_close.pct_change()
                feats['NIFTY_Return_5d'] = n_close.pct_change(5)
                feats['NIFTY_Vol_20'] = feats['NIFTY_Return_1d'].rolling(20).std()
                n_sma50 = n_close.rolling(50).mean()
                feats['NIFTY_Dist_SMA50'] = (n_close / n_sma50) - 1
            
            if 'VIX' in context_dfs and context_dfs['VIX'] is not None and not context_dfs['VIX'].empty:
                v_close = context_dfs['VIX']['Close'].reindex(df.index)
                feats['VIX_Level'] = v_close / 100.0
                feats['VIX_Change_1d'] = v_close.pct_change()
        
        # Clean infinities and warm-up NaNs (due to SMA200 lookback)
        feats = feats.replace([np.inf, -np.inf], np.nan)
        df_clean = feats.dropna()
        dropped = initial_rows - len(df_clean)
        
        logger.info(f"Total stationary features generated: {len(df_clean.columns)}")
        logger.info(f"Warm-up handling: {dropped} initial rows removed (e.g. 200 SMA period).")
        logger.info(f"Missing values remaining in feature matrix: {df_clean.isna().sum().sum()}")
        
        # Guard against duplicate columns
        if df_clean.columns.duplicated().any():
            dup_cols = df_clean.columns[df_clean.columns.duplicated()].tolist()
            raise ValueError(f"Duplicate columns detected: {dup_cols}")
            
        return df_clean
