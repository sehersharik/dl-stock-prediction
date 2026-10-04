import pandas as pd
import json
import logging
from datetime import datetime
from src.utils.config import PROJECT_ROOT
from src.data.provider import MarketDataProvider

logger = logging.getLogger(__name__)

class DataIngester:
    """Orchestrates fetching, validation, and storage of raw historical data."""
    
    def __init__(self, provider: MarketDataProvider, config: dict):
        self.provider = provider
        self.config = config
        self.raw_dir = PROJECT_ROOT / self.config['data'].get('raw_dir', 'data/raw')
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def validate_and_clean(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        if df.empty:
            raise ValueError(f"Empty dataframe received for {ticker}.")
        
        # 1. Expected Columns Check
        expected_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [c for c in expected_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns for {ticker}: {missing_cols}")
            
        # 2. Datetime Index Verification
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
            
        # 3. Timezone Normalization
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
            
        # 4. Sorting & Chronological Verification
        df = df.sort_index()
        
        # 5. Duplicate Detection & Handling
        if df.index.duplicated().any():
            duplicate_count = df.index.duplicated().sum()
            logger.warning(f"Found {duplicate_count} duplicate timestamps for {ticker}. Keeping first occurrences.")
            df = df[~df.index.duplicated(keep='first')]
            
        return df

    def generate_report(self, df: pd.DataFrame, ticker: str) -> dict:
        report = {
            "number_of_rows": len(df),
            "start_date": str(df.index.min().date()),
            "end_date": str(df.index.max().date()),
            "missing_values": df.isnull().sum().to_dict(),
            "duplicate_rows": int(df.index.duplicated().sum()), # Checked before filtering logically, but 0 here since handled
            "frequency": pd.infer_freq(df.index) or "Unknown/Irregular"
        }
        
        logger.info(f"=== Data Validation Report for {ticker} ===")
        for k, v in report.items():
            logger.info(f"  {k}: {v}")
        return report

    def ingest(self, ticker: str, start_date: str, end_date: str, interval: str = "1d"):
        try:
            # Fetch
            df = self.provider.fetch_historical(ticker, start_date, end_date, interval)
            
            # Validate
            df = self.validate_and_clean(df, ticker)
            
            # Report
            report = self.generate_report(df, ticker)
            
            # Output Paths
            filename_base = f"{ticker}_{start_date}_{end_date}"
            csv_path = self.raw_dir / f"{filename_base}.csv"
            meta_path = self.raw_dir / f"{filename_base}_meta.json"
            
            # Save Data
            df.to_csv(csv_path)
            
            # Save Metadata
            meta = {
                "ticker": ticker,
                "source": self.provider.__class__.__name__,
                "download_timestamp": datetime.now().isoformat(),
                "interval": interval,
                "requested_date_range": {"start": start_date, "end": end_date},
                "actual_date_range": {"start": report["start_date"], "end": report["end_date"]},
                "validation_report": report
            }
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=4)
                
            logger.info(f"Successfully saved {ticker} data to {csv_path}")
            
        except Exception as e:
            logger.error(f"Ingestion pipeline failed for {ticker}: {e}")
            raise e
