from abc import ABC, abstractmethod
import pandas as pd
import yfinance as yf
import logging
import time

logger = logging.getLogger(__name__)

class MarketDataProvider(ABC):
    """Abstract interface for fetching market data, avoiding hard-coupling to yfinance."""
    
    @abstractmethod
    def fetch_historical(self, ticker: str, start_date: str, end_date: str, interval: str = "1d") -> pd.DataFrame:
        """Fetches historical OHLCV data."""
        pass

class YFinanceHistoricalProvider(MarketDataProvider):
    """yfinance implementation of the MarketDataProvider with robust retry logic."""
    
    def __init__(self, retries: int = 3, backoff_factor: int = 2):
        self.retries = retries
        self.backoff_factor = backoff_factor

    def fetch_historical(self, ticker: str, start_date: str, end_date: str, interval: str = "1d") -> pd.DataFrame:
        attempt = 0
        while attempt < self.retries:
            try:
                logger.info(f"Fetching {ticker} from {start_date} to {end_date} (Attempt {attempt+1})")
                
                # Fetch data
                df = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)
                
                if df is None or df.empty:
                    raise ValueError(f"Provider returned empty data for {ticker}.")
                
                # Handle yfinance multi-index columns which occurs on some versions/requests
                if isinstance(df.columns, pd.MultiIndex):
                    # Usually level 0 is price fields (Open, High), level 1 is Ticker
                    df.columns = df.columns.get_level_values(0)
                
                return df
                
            except Exception as e:
                attempt += 1
                logger.warning(f"Failed to fetch {ticker}: {e}")
                if attempt >= self.retries:
                    logger.error(f"Exhausted retries for {ticker}.")
                    raise e
                time.sleep(self.backoff_factor ** attempt)

import time
import os
from abc import ABC, abstractmethod

class LiveDataProvider(ABC):
    """
    Abstract interface for retrieving live market data.
    Must support fetching sufficient context (lookback) to calculate features.
    """
    @abstractmethod
    def get_latest_data(self, ticker: str, lookback_days: int = 250) -> dict:
        pass

class YFinanceLiveProvider(LiveDataProvider):
    """
    Safest initial implementation using Yahoo Finance.
    NOTE: YFinance data for NSE (India) is typically delayed by 15 minutes.
    Does not require API keys, making it robust for testing without credentials.
    """
    def __init__(self, retries: int = 3, delay_secs: int = 5):
        self.retries = retries
        self.delay_secs = delay_secs
        
    def get_latest_data(self, ticker: str, lookback_days: int = 250) -> dict:
        import yfinance as yf
        from datetime import datetime, timedelta
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        for attempt in range(self.retries):
            try:
                df = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), progress=False)
                if df.empty:
                    raise ValueError(f"No data returned for {ticker}.")
                
                # Format to match historical expectations
                df.index = df.index.tz_localize(None)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)
                    
                df = df.rename(columns=lambda x: x.capitalize() if x != 'Adj Close' else x)
                
                # Check stale data
                last_timestamp = df.index[-1]
                time_diff = end_date - last_timestamp
                is_stale = time_diff > timedelta(days=2) # e.g., weekends
                
                return {
                    "data": df,
                    "timestamp": datetime.now().isoformat(),
                    "last_data_timestamp": last_timestamp.isoformat(),
                    "source": "Yahoo Finance (REST)",
                    "interval": "1d",
                    "status": "DELAYED (15 min)",
                    "stale_warning": is_stale
                }
                
            except Exception as e:
                logger.warning(f"Live data fetch failed (Attempt {attempt+1}/{self.retries}): {e}")
                if attempt < self.retries - 1:
                    time.sleep(self.delay_secs)
                else:
                    raise ConnectionError(f"Failed to fetch live data after {self.retries} attempts.")
