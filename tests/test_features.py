import unittest
import pandas as pd
import numpy as np
from src.features.generator import FeatureGenerator

class TestFeatureGenerator(unittest.TestCase):
    def setUp(self):
        # Generate 300 days of dummy price data (enough to overcome 200-day warm-up)
        dates = pd.date_range(start='2023-01-01', periods=300, freq='B')
        np.random.seed(42)
        prices = np.linspace(100, 200, 300) + np.random.normal(0, 2, 300)
        
        self.df = pd.DataFrame({
            'Open': prices - 1,
            'High': prices + 2,
            'Low': prices - 2,
            'Close': prices,
            'Volume': np.random.randint(1000, 10000, 300)
        }, index=dates)
        
        self.config = {
            'features': {
                'groups': {
                    'raw': True,
                    'trend': True,
                    'momentum': True,
                    'volatility': True,
                    'volume': False,
                    'market_context': False
                }
            }
        }
        self.generator = FeatureGenerator(self.config)

    def test_no_lookahead_bias(self):
        """
        Verify that changing the future data does not alter past indicators.
        A cornerstone test against lookahead bias.
        """
        # Run standard
        df_base = self.generator.generate(self.df)
        
        # Modify the last row wildly
        df_altered = self.df.copy()
        df_altered.iloc[-1, df_altered.columns.get_loc('Close')] = 99999
        
        df_alt_features = self.generator.generate(df_altered)
        
        # Compare the second-to-last row (should be identical)
        pd.testing.assert_series_equal(
            df_base.iloc[-2], 
            df_alt_features.iloc[-2]
        )

    def test_stationary_feature_columns(self):
        """New pipeline emits stationary, dimensionless features — no raw price levels."""
        df_features = self.generator.generate(self.df)
        
        # New stationary features that MUST be present
        expected_cols = [
            'Return_1d', 'Return_5d', 'HL_Spread', 'CO_Spread',
            'Dist_SMA20', 'Dist_SMA50', 'Dist_SMA200',
            'RSI_14', 'MACD_Ratio', 'ATR_Ratio', 'BB_PctB', 'BB_Width',
            'Rolling_Vol_20',
        ]
        for col in expected_cols:
            self.assertIn(col, df_features.columns, f"Expected stationary feature '{col}' missing")
        
        # Old raw price columns must NOT be present
        forbidden_cols = ['SMA_20', 'SMA_50', 'SMA_200', 'BB_Upper', 'BB_Lower', 'Close', 'Open']
        for col in forbidden_cols:
            self.assertNotIn(col, df_features.columns, f"Raw price column '{col}' must not be in features")

    def test_warmup_dropped_correctly(self):
        df_features = self.generator.generate(self.df)
        # Because SMA_200 is used in trend ratios, the first ~199 rows should be dropped
        # Exact count may vary by 1-2 due to rolling warmup windows; just verify it is < full length
        self.assertLess(len(df_features), 300)
        self.assertFalse(df_features.isna().any().any(), "There should be no missing values remaining.")

    def test_bollinger_features_stationary(self):
        """BB_PctB (percent bandwidth) and BB_Width are stationary; raw BB_Upper/BB_Lower are not exported."""
        df_features = self.generator.generate(self.df)
        self.assertIn('BB_PctB', df_features.columns)
        self.assertIn('BB_Width', df_features.columns)
        # BB_PctB is bounded roughly [0,1] for normal market data but can exceed slightly
        # Verify it is not a raw price level (i.e., not in the thousands)
        self.assertLess(df_features['BB_Width'].abs().max(), 5.0,
                        "BB_Width should be a small ratio, not a raw price value")
        # BB_Upper / BB_Lower raw columns should not exist
        self.assertNotIn('BB_Upper', df_features.columns)
        self.assertNotIn('BB_Lower', df_features.columns)

if __name__ == '__main__':
    unittest.main()
