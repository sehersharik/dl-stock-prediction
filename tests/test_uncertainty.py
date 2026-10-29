import unittest

import numpy as np
from sklearn.metrics import roc_auc_score

from src.evaluation.uncertainty import bootstrap, circular_shift_test, roc_auc, stationary_bootstrap_indices


class TestUncertainty(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.y = rng.integers(0, 2, 300)
        self.noise = rng.random(300)

    def test_rank_auc_matches_sklearn_with_ties(self):
        p = np.round(self.noise, 1)
        self.assertAlmostEqual(roc_auc(self.y, p), roc_auc_score(self.y, p), places=12)

    def test_perfect_predictor_excludes_chance(self):
        p = self.y * 0.5 + 0.25
        out = bootstrap(self.y, {"perfect": (p, self.y)}, reference="perfect", n_boot=300)
        self.assertGreater(out["models"]["perfect"]["ROC-AUC"]["ci95"][0], 0.5)
        self.assertTrue(out["models"]["perfect"]["ROC-AUC"]["excludes_null"])
        self.assertLess(circular_shift_test(self.y, p, self.y)["ROC-AUC"]["p_value"], 0.01)

    def test_noise_predictor_includes_chance(self):
        pred = (self.noise >= 0.5).astype(int)
        out = bootstrap(self.y, {"noise": (self.noise, pred)}, reference="noise", n_boot=500)
        lo, hi = out["models"]["noise"]["ROC-AUC"]["ci95"]
        self.assertLess(lo, 0.5)
        self.assertGreater(hi, 0.5)

    def test_identical_models_have_zero_paired_difference(self):
        pred = (self.noise >= 0.5).astype(int)
        out = bootstrap(self.y, {"a": (self.noise, pred), "b": (self.noise.copy(), pred.copy())}, reference="a", n_boot=200)
        self.assertEqual(out["differences"]["a - b"]["ROC-AUC"]["ci95"], [0.0, 0.0])

    def test_stationary_bootstrap_mean_block_length(self):
        rng = np.random.default_rng(0)
        idx = np.stack([stationary_bootstrap_indices(500, 10, rng) for _ in range(400)])
        steps = np.diff(idx, axis=1)
        continues = (steps == 1) | (steps == -499)
        self.assertAlmostEqual(continues.size / (continues.size - continues.sum()), 10, delta=0.5)


if __name__ == '__main__':
    unittest.main()
