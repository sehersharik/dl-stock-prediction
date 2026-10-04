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

    def test_sma_calculation(self):
        df_features = self.generator.generate(self.df)
        
        # Check standard pandas SMA manually against the returned feature
        expected_sma_20 = self.df['Close'].rolling(20).mean().dropna()
        # Features index must align perfectly
        pd.testing.assert_series_equal(
            df_features['SMA_20'], 
            expected_sma_20.loc[df_features.index],
            check_names=False
        )

    def test_warmup_dropped_correctly(self):
        df_features = self.generator.generate(self.df)
        # Because SMA_200 is enabled in 'trend', the first 199 rows should be dropped
        self.assertEqual(len(df_features), 300 - 199)
        self.assertFalse(df_features.isna().any().any(), "There should be no missing values remaining.")

    def test_bollinger_bands(self):
        df_features = self.generator.generate(self.df)
        self.assertIn('BB_Upper', df_features.columns)
        self.assertIn('BB_Lower', df_features.columns)
        # Upper > Lower
        self.assertTrue((df_features['BB_Upper'] >= df_features['BB_Lower']).all())

if __name__ == '__main__':
    unittest.main()
