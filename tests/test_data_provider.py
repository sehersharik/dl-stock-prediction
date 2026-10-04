import unittest
import pandas as pd
from src.data.provider import MarketDataProvider
from src.data.fetcher import DataIngester

class MockProvider(MarketDataProvider):
    """A deterministic mock provider for testing validation rules."""
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
    def fetch_historical(self, ticker: str, start_date: str, end_date: str, interval: str = "1d") -> pd.DataFrame:
        return self.df

class TestDataIngester(unittest.TestCase):
    def setUp(self):
        self.config = {"data": {"raw_dir": "data/raw"}}
        
    def test_empty_data_raises_error(self):
        df = pd.DataFrame()
        provider = MockProvider(df)
        ingester = DataIngester(provider, self.config)
        with self.assertRaises(ValueError) as context:
            ingester.validate_and_clean(df, "TEST")
        self.assertIn("Empty dataframe", str(context.exception))
            
    def test_missing_columns_raises_error(self):
        df = pd.DataFrame({"Open": [100], "Close": [101]})
        provider = MockProvider(df)
        ingester = DataIngester(provider, self.config)
        with self.assertRaises(ValueError) as context:
            ingester.validate_and_clean(df, "TEST")
        self.assertIn("Missing required columns", str(context.exception))
            
    def test_duplicates_and_sorting_handled(self):
        # Out of order dates with a duplicate
        dates = pd.to_datetime(["2024-01-03", "2024-01-01", "2024-01-01"])
        df = pd.DataFrame({
            "Open": [3, 1, 99],
            "High": [3, 1, 99],
            "Low": [3, 1, 99],
            "Close": [3, 1, 99],
            "Volume": [3, 1, 99]
        }, index=dates)
        
        provider = MockProvider(df)
        ingester = DataIngester(provider, self.config)
        cleaned = ingester.validate_and_clean(df, "TEST")
        
        # 1. Ensure sorted
        self.assertTrue(cleaned.index.is_monotonic_increasing)
        
        # 2. Ensure duplicate dropped (keeps first occurrence)
        self.assertEqual(len(cleaned), 2)
        
        # 3. Ensure correct value retained for the duplicate index
        self.assertEqual(cleaned.loc["2024-01-01", "Open"], 1)

if __name__ == '__main__':
    unittest.main()
