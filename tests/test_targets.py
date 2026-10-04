import unittest
import pandas as pd
from src.targets.generator import TargetGenerator

class TestTargetGenerator(unittest.TestCase):
    def setUp(self):
        self.config = {
            'targets': {
                'downside_risk_threshold': -0.02
            }
        }
        self.generator = TargetGenerator(self.config)
        
        # 5 days of data
        dates = pd.date_range('2024-01-01', periods=5, freq='B')
        self.df = pd.DataFrame({
            'Close': [100.0, 105.0, 100.0, 95.0, 96.0],
            'Volume': [10, 10, 10, 10, 10]
        }, index=dates)

    def test_target_alignment(self):
        """Tests that T's target accurately reflects T+1's return."""
        X, y = self.generator.generate(self.df, drop_na_targets=True)
        
        # Day 1 (100 -> 105) = +5%
        self.assertAlmostEqual(y.iloc[0]['Next_Day_Return'], 0.05)
        self.assertEqual(y.iloc[0]['Target_Direction'], 1)
        self.assertEqual(y.iloc[0]['Target_Risk'], 0)
        
        # Day 2 (105 -> 100) = -4.76% (Breaks -0.02 threshold)
        self.assertAlmostEqual(y.iloc[1]['Next_Day_Return'], -0.047619, places=5)
        self.assertEqual(y.iloc[1]['Target_Direction'], 0)
        self.assertEqual(y.iloc[1]['Target_Risk'], 1)
        
        # Last day should be removed due to NaN future return
        self.assertEqual(len(X), 4)
        self.assertEqual(len(y), 4)
        
    def test_leakage_prevention(self):
        """Simulate a leak and ensure the system halts immediately."""
        df_leaked = self.df.copy()
        df_leaked['Next_Day_Return'] = 0.1 # Artificial leak
        
        with self.assertRaises(ValueError) as context:
            self.generator.generate(df_leaked)
            
        self.assertIn("CRITICAL: Target leakage detected", str(context.exception))
        
    def test_index_alignment(self):
        """X and y must remain perfectly aligned."""
        X, y = self.generator.generate(self.df, drop_na_targets=True)
        self.assertTrue(X.index.equals(y.index))

if __name__ == '__main__':
    unittest.main()
