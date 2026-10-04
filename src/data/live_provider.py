import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
import logging
import time

logger = logging.getLogger(__name__)

class LiveDataProvider:
    def get_latest_data(self, ticker: str):
        pass
    def get_lookback_data(self, ticker: str, lookback_days: int):
        pass

class YFinanceLiveProvider(LiveDataProvider):
    def __init__(self, retries=3, stale_threshold_minutes=30):
        self.retries = retries
        self.stale_threshold = timedelta(minutes=stale_threshold_minutes)
        
    def get_lookback_data(self, ticker: str, lookback_days: int = 250) -> pd.DataFrame:
        """Fetches sufficient lookback data to accurately compute features."""
        for attempt in range(self.retries):
            try:
                tkr = yf.Ticker(ticker)
                # '1y' covers roughly 252 trading days, which is enough for a 200 SMA
                df = tkr.history(period='1y')
                if df.empty:
                    raise ValueError(f"No lookback data returned for {ticker}")
                
                # Convert timezone to naive local time (Asia/Kolkata)
                if df.index.tzinfo is not None:
                    df.index = df.index.tz_convert('Asia/Kolkata').tz_localize(None)
                
                return df[['Open', 'High', 'Low', 'Close', 'Volume']].tail(lookback_days)
            except Exception as e:
                logger.warning(f"Live lookback fetch attempt {attempt+1} failed: {e}")
                time.sleep(2)
        raise ConnectionError(f"Failed to fetch live lookback data for {ticker}")

    def get_latest_data(self, ticker: str) -> dict:
        for attempt in range(self.retries):
            try:
                tkr = yf.Ticker(ticker)
                df = tkr.history(period='1d', interval='1m')
                
                if df.empty:
                    df = tkr.history(period='1d')
                    if df.empty:
                        raise ValueError(f"No data returned for {ticker}")
                        
                latest_row = df.iloc[-1]
                timestamp = df.index[-1]
                
                if timestamp.tzinfo is not None:
                    timestamp = timestamp.tz_convert('Asia/Kolkata').tz_localize(None)
                
                now = datetime.now()
                is_stale = (now - timestamp) > self.stale_threshold
                
                return {
                    'timestamp': timestamp,
                    'Open': float(latest_row['Open']),
                    'High': float(latest_row['High']),
                    'Low': float(latest_row['Low']),
                    'Close': float(latest_row['Close']),
                    'Volume': float(latest_row['Volume']),
                    'status': 'Stale/Delayed' if is_stale else 'Delayed Snapshot',
                    'source': 'yfinance',
                    'interval': '1m/1d fallback'
                }
            except Exception as e:
                logger.warning(f"Live fetch attempt {attempt+1} failed: {e}")
                time.sleep(2)
        raise ConnectionError(f"Failed to fetch live data for {ticker} after {self.retries} attempts.")
