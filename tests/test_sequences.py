import unittest

import numpy as np
import pandas as pd
import torch

from src.models.dl_models import ConfigurableLSTM
from src.models.dl_trainer import DLTrainer


class TestSequences(unittest.TestCase):
    def test_windows_end_on_the_labelled_day_and_only_look_back(self):
        idx = pd.bdate_range("2025-01-01", periods=8)
        X = pd.DataFrame({"f": np.arange(8.0)}, index=idx)
        y = pd.Series(np.arange(8) % 2, index=idx)
        Xs, ys, ends = DLTrainer._create_sequences(None, X, y, seq_length=3)
        self.assertEqual(len(Xs), 6)
        self.assertEqual(pd.Timestamp(ends[0]), idx[2])
        np.testing.assert_array_equal(Xs[0, :, 0], [0.0, 1.0, 2.0])  # rows t-2..t, nothing after t
        self.assertEqual(ys[0], y.iloc[2])

    def test_gru_and_lstm_cells_produce_one_logit_per_window(self):
        x = torch.randn(4, 10, 27)
        for cell in ("lstm", "gru"):
            self.assertEqual(tuple(ConfigurableLSTM(27, cell=cell)(x).shape), (4, 1))


if __name__ == '__main__':
    unittest.main()
