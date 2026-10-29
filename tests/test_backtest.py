import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtesting.engine import Backtester

COST = 0.0015  # 0.10% transaction cost + 0.05% slippage, per side


def make_bt(rf=0.0):
    cfg = {"data": {"raw_dir": "data/raw"},
           "backtesting": {"transaction_cost": 0.001, "slippage": 0.0005, "risk_free_rate_annual": rf}}
    return Backtester(cfg, Path(__file__).resolve().parent.parent)


class TestBacktester(unittest.TestCase):
    def setUp(self):
        self.idx = pd.bdate_range("2025-01-01", periods=6)
        self.prices = pd.DataFrame({"Open": [100, 102, 101, 105, 104, 108],
                                    "Close": [101, 103, 100, 106, 103, 110]}, index=self.idx, dtype=float)

    def preds(self, probs, dates=None):
        return pd.DataFrame({"Prob": probs}, index=self.idx[:len(probs)] if dates is None else dates)

    def test_period_is_next_session_even_when_prediction_rows_are_missing(self):
        # Predictions skip session 2; the return for session 1 must still be close1 -> close2
        dates = self.idx[[0, 1, 3]]
        df = make_bt().simulate(self.preds([0.9, 0.9, 0.9], dates), self.prices, 0.5, "close_to_close")
        self.assertAlmostEqual(df.loc[self.idx[1], "Period_Return"], 100 / 103 - 1)
        self.assertAlmostEqual(df.loc[self.idx[3], "Period_Return"], 103 / 106 - 1)

    def test_next_open_execution_uses_open_t1_to_open_t2(self):
        df = make_bt().simulate(self.preds([0.9] * 4), self.prices, 0.5, "next_open")
        self.assertAlmostEqual(df.loc[self.idx[0], "Period_Return"], 101 / 102 - 1)
        self.assertAlmostEqual(df.loc[self.idx[3], "Period_Return"], 108 / 104 - 1)

    def test_rows_without_a_complete_next_period_are_dropped(self):
        df = make_bt().simulate(self.preds([0.9] * 6), self.prices, 0.5, "next_open")
        self.assertEqual(len(df), 4)  # last two sessions have no open T+2

    def test_costs_are_charged_per_side_including_final_exit(self):
        # long, long, cash, long  -> entry, exit, entry, final exit = 4 sides
        df = make_bt().simulate(self.preds([0.9, 0.9, 0.1, 0.9]), self.prices, 0.5, "close_to_close")
        np.testing.assert_allclose(df["Friction"].values, [COST, 0, COST, 2 * COST])
        r = df["Period_Return"].values
        np.testing.assert_allclose(df["Strategy_Return"].values, [r[0] - COST, r[1], -COST, r[3] - 2 * COST])
        m = make_bt()._calculate_metrics(df)
        self.assertEqual(m["Position_Changes"], 4)
        self.assertEqual(m["Trades"], 2)
        self.assertAlmostEqual(m["Total_Cost_Paid"], 4 * COST)

    def test_buy_and_hold_uses_same_periods_and_pays_entry_and_exit(self):
        df = make_bt().simulate(self.preds([0.1, 0.1, 0.1, 0.1]), self.prices, 0.5, "close_to_close")
        expected = (1 + df["Period_Return"]).prod() * (1 - COST) ** 0  # gross
        bnh_total = (1 + df["BnH_Return"]).prod() - 1
        r = df["Period_Return"].values
        manual = (1 + r[0] - COST) * (1 + r[1]) * (1 + r[2]) * (1 + r[3] - COST) - 1
        self.assertAlmostEqual(bnh_total, manual)
        self.assertLess(bnh_total, expected - 1)
        self.assertTrue((df["Strategy_Return"] == 0).all())  # always in cash, never trades

    def test_per_trade_win_rate_includes_exit_cost(self):
        # One long period earning less than the round-trip cost is a losing trade
        prices = pd.DataFrame({"Open": [100.0] * 4, "Close": [100.0, 100.2, 100.2, 100.2]},
                              index=self.idx[:4])
        df = make_bt().simulate(self.preds([0.9, 0.1, 0.1]), prices, 0.5, "close_to_close")
        m = make_bt()._calculate_metrics(df)
        self.assertEqual(m["Trades"], 1)
        self.assertAlmostEqual(make_bt().trade_returns(df)[0], (1 + 0.002 - COST) * (1 - COST) - 1)
        self.assertEqual(m["Win_Rate_Trade"], 0.0)  # +0.20% gross < 0.30% round-trip cost
        self.assertEqual(m["Win_Rate_Day"], 1.0)  # the in-market day itself was positive

    def test_risk_free_rate_lowers_sharpe(self):
        df = make_bt().simulate(self.preds([0.9] * 5), self.prices, 0.5, "close_to_close")
        m0 = make_bt(rf=0.0)._calculate_metrics(df)
        m6 = make_bt(rf=0.06)._calculate_metrics(df)
        self.assertAlmostEqual(m0["Strategy_Sharpe"], m0["Strategy_Sharpe_rf0"])
        self.assertLess(m6["Strategy_Sharpe"], m0["Strategy_Sharpe"])

    def test_max_drawdown_matches_equity_curve(self):
        df = make_bt().simulate(self.preds([0.9] * 5), self.prices, 0.5, "close_to_close")
        eq = df["Strategy_Equity"]
        self.assertAlmostEqual(df["Strategy_Drawdown"].min(), (eq / eq.cummax() - 1).min())

    def test_band_holds_position_between_thresholds(self):
        sig = Backtester.signals(pd.Series([0.50, 0.56, 0.52, 0.48, 0.44, 0.50, 0.55]), 0.55, 0.45)
        self.assertEqual(sig.tolist(), [0, 1, 1, 1, 0, 0, 1])

    def test_plain_threshold_is_unchanged_by_band_logic(self):
        prob = pd.Series([0.49, 0.50, 0.51, 0.3, 0.7])
        self.assertEqual(Backtester.signals(prob, 0.5, 0.5).tolist(), (prob >= 0.5).astype(int).tolist())


if __name__ == '__main__':
    unittest.main()
