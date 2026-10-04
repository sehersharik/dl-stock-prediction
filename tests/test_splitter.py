import unittest
import pandas as pd
import numpy as np
from src.evaluation.splitter import TimeSeriesSplitter

class TestTimeSeriesSplitter(unittest.TestCase):
    def setUp(self):
        self.config = {
            'training': {
                'train_split': 0.7,
                'val_split': 0.15
            }
        }
        self.splitter = TimeSeriesSplitter(self.config)
        
        # Generate 100 days of dummy data
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        self.X = pd.DataFrame({'feature1': np.random.rand(100)}, index=dates)
        self.y = pd.DataFrame({'target': np.random.randint(0, 2, 100)}, index=dates)

    def test_chronological_splits(self):
        """Verifies no overlap between Train, Validation, and Test."""
        (X_train, _), (X_val, _), (X_test, _) = self.splitter.split_train_val_test(self.X, self.y)
        
        self.assertEqual(len(X_train), 70)
        self.assertEqual(len(X_val), 15)
        self.assertEqual(len(X_test), 15)
        
        # Train comes strictly before Val
        self.assertLess(X_train.index.max(), X_val.index.min())
        # Val comes strictly before Test
        self.assertLess(X_val.index.max(), X_test.index.min())

    def test_strict_scaler(self):
        """Verifies scaler is only fit on training data."""
        (X_train, _), (X_val, _), (X_test, _) = self.splitter.split_train_val_test(self.X, self.y)
        scaler, X_tr_s, X_v_s, X_te_s = self.splitter.scale_features(X_train, X_val, X_test)
        
        # The mean of the scaled training data should be virtually 0 
        # (because it was explicitly fitted on it)
        self.assertAlmostEqual(X_tr_s['feature1'].mean(), 0.0, places=5)
        
        # But the validation data mean will NOT be perfectly 0, proving 
        # the scaler was not leaked/fitted on it.
        self.assertNotAlmostEqual(X_v_s['feature1'].mean(), 0.0, places=5)

    def test_leakage_assertion(self):
        """Mocks an overlapped timeframe to ensure validation trips."""
        overlap_train = self.X.iloc[:50]
        overlap_val = self.X.iloc[45:60] # Overlaps indices 45 to 50!
        
        with self.assertRaises(AssertionError) as context:
            self.splitter.validate_chronology(overlap_train, overlap_val)
        self.assertIn("Leakage", str(context.exception))

if __name__ == '__main__':
    unittest.main()
