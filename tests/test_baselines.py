import unittest
import numpy as np
from src.models.baselines import get_baseline_model

class TestBaselines(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # 100 samples, 5 features
        self.X = np.random.rand(100, 5)
        # Binary target
        self.y = np.random.randint(0, 2, 100)
        
    def test_majority_baseline_shapes(self):
        model = get_baseline_model('majority')
        model.fit(self.X, self.y)
        
        preds = model.predict(self.X)
        probs = model.predict_proba(self.X)
        
        self.assertEqual(preds.shape, (100,))
        self.assertEqual(probs.shape, (100, 2))
        
        # The majority class should be 1 or 0, prob should be 1.0 for it.
        majority = np.bincount(self.y).argmax()
        self.assertTrue((preds == majority).all())
        
    def test_logistic_shapes(self):
        model = get_baseline_model('logistic')
        model.fit(self.X, self.y)
        
        preds = model.predict(self.X)
        probs = model.predict_proba(self.X)
        
        self.assertEqual(preds.shape, (100,))
        self.assertEqual(probs.shape, (100, 2))
        self.assertTrue(0 <= probs.min() and probs.max() <= 1)

if __name__ == '__main__':
    unittest.main()
