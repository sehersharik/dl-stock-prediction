import unittest

import numpy as np

from src.evaluation.calibration import brier, log_loss, reliability_bins, summarize, wilson_interval


class TestCalibration(unittest.TestCase):
    def test_coin_flip_scores(self):
        y = np.array([0, 1] * 50, dtype=float)
        p = np.full(100, 0.5)
        self.assertAlmostEqual(brier(y, p), 0.25)
        self.assertAlmostEqual(log_loss(y, p), np.log(2))

    def test_skill_score_is_zero_for_the_reference_forecast(self):
        y = np.random.default_rng(0).integers(0, 2, 200).astype(float)
        s = summarize(y, np.full(200, 0.4), reference_rate=0.4)
        self.assertAlmostEqual(s["brier_skill_score"], 0.0)

    def test_equal_count_bins_cover_every_observation(self):
        rng = np.random.default_rng(1)
        p, y = rng.random(203), rng.integers(0, 2, 203)
        self.assertEqual(sum(b["n"] for b in reliability_bins(y, p, 5)), 203)

    def test_wilson_interval_contains_estimate(self):
        lo, hi = wilson_interval(30, 57)
        self.assertLess(lo, 30 / 57)
        self.assertGreater(hi, 30 / 57)


if __name__ == '__main__':
    unittest.main()
