import unittest
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from src.data.live_provider import LiveDataProvider
from src.inference.engine import InferenceEngine
from src.utils.config import load_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
TICKER = "RELIANCE.NS"
MODEL = "mlp"


def _read_raw(name: str) -> pd.DataFrame:
    return pd.read_csv(RAW_DIR / name, index_col=0, parse_dates=True)


class MockLiveProvider(LiveDataProvider):
    """Deterministic provider serving committed raw CSVs instead of yfinance."""
    def __init__(self, history: pd.DataFrame, context: dict, tick_time: datetime):
        self.history = history
        self.context = context
        self.tick_time = tick_time

    def get_lookback_data(self, ticker: str, lookback_days: int = 260) -> pd.DataFrame:
        return self.history.copy()

    def get_context_data(self, context_tickers: dict, lookback_days: int = 260) -> dict:
        return {k: v.copy() for k, v in self.context.items()}

    def get_latest_data(self, ticker: str) -> dict:
        # A deliberately unrealistic 1-minute bar: if it leaked into the features,
        # HL_Spread / CO_Spread / Vol_Ratio_20 would change.
        return {
            'timestamp': pd.Timestamp(self.tick_time),
            'Open': 1000.0, 'High': 5000.0, 'Low': 1.0, 'Close': 1200.0, 'Volume': 1.0,
            'status': 'Delayed Snapshot', 'source': 'mock', 'interval': '1m',
        }


class TestLiveInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config("settings.yaml")
        cls.reliance = _read_raw("RELIANCE.NS_10y.csv")
        cls.nifty = _read_raw("NSEI_10y.csv")
        cls.vix = _read_raw("INDIAVIX_10y.csv")
        cls.X_train_schema = pd.read_csv(
            PROJECT_ROOT / "data" / "processed" / f"{TICKER}_X.csv", index_col=0, parse_dates=True
        )

    def _run(self, end_date: str, now: datetime, context: dict = None, lookback_rows: int = 500):
        history = self.reliance.loc[:end_date].tail(lookback_rows)
        if context is None:
            context = {"NIFTY": self.nifty.loc[:end_date], "VIX": self.vix.loc[:end_date]}
        provider = MockLiveProvider(history, context, tick_time=now)
        engine = InferenceEngine(self.config, provider, PROJECT_ROOT)
        return engine.run_live_inference(TICKER, MODEL, now=now)

    # Fixture dates end at 2026-09-11: the committed raw CSVs contain zero-volume RELIANCE
    # placeholder bars on NSE holidays (2026-09-14, 2026-10-02) with no NIFTY/VIX bar, which
    # blanks the 50-day NIFTY window for later dates (covered by the holiday test below).

    def test_valid_market_data_produces_full_schema_in_training_order(self):
        result = self._run("2026-09-11", now=datetime(2026, 9, 11, 17, 0))
        self.assertEqual(result["status"], "Success", result.get("message"))
        self.assertEqual(list(result["latest_features"].keys()), self.X_train_schema.columns.tolist())
        self.assertEqual(len(result["latest_features"]), 27)
        self.assertTrue(0.0 < result["direction_prob"] < 1.0)

    def test_live_features_match_training_features(self):
        """Parity: live pipeline reproduces the training feature row for the same session."""
        session = self.X_train_schema.index[-1]
        result = self._run(str(session.date()), now=datetime.combine(session.date(), datetime.min.time()).replace(hour=18))
        self.assertEqual(result["status"], "Success", result.get("message"))
        self.assertEqual(result["feature_date"], session)
        live = pd.Series(result["latest_features"])
        trained = self.X_train_schema.loc[session]
        np.testing.assert_allclose(live.values, trained.values, rtol=1e-6, atol=1e-8)

    def test_intraday_tick_and_unfinished_session_not_used(self):
        # During market hours today's daily bar is incomplete -> use previous session.
        during = self._run("2026-09-11", now=datetime(2026, 9, 11, 12, 0))
        self.assertEqual(during["status"], "Success", during.get("message"))
        self.assertEqual(during["feature_date"], pd.Timestamp("2026-09-10"))

        # The same session evaluated the next day must give identical features,
        # proving the 1-minute tick did not enter the feature row. (Tolerance covers EWM
        # warm-up from a lookback window starting one row later; a leaked tick would be far larger.)
        reference = self._run("2026-09-10", now=datetime(2026, 9, 11, 9, 0))
        self.assertEqual(reference["feature_date"], pd.Timestamp("2026-09-10"))
        np.testing.assert_allclose(
            pd.Series(during["latest_features"]).values,
            pd.Series(reference["latest_features"]).values, rtol=1e-8, atol=1e-10
        )

        # After the session completes, today's bar becomes the feature row.
        after = self._run("2026-09-11", now=datetime(2026, 9, 11, 17, 0))
        self.assertEqual(after["feature_date"], pd.Timestamp("2026-09-11"))

    def test_missing_market_context_returns_controlled_error(self):
        result = self._run("2026-09-11", now=datetime(2026, 9, 11, 17, 0), context={"NIFTY": self.nifty})
        self.assertEqual(result["status"], "Error")
        self.assertIn("market context", result["message"])
        self.assertIn("VIX", result["message"])

    def test_context_missing_for_latest_session_is_not_silently_replaced(self):
        context = {"NIFTY": self.nifty.loc[:"2026-09-10"], "VIX": self.vix.loc[:"2026-09-11"]}
        result = self._run("2026-09-11", now=datetime(2026, 9, 11, 17, 0), context=context)
        self.assertEqual(result["status"], "Error")
        self.assertIn("2026-09-11", result["message"])

    def test_holiday_placeholder_bar_withholds_prediction(self):
        """A RELIANCE holiday bar with no NIFTY/VIX bar (2026-10-02) leaves the context
        windows incomplete; the prediction must be withheld, not filled in."""
        result = self._run("2026-10-06", now=datetime(2026, 10, 6, 17, 0))
        self.assertEqual(result["status"], "Error")
        self.assertIn("2026-10-06", result["message"])


if __name__ == '__main__':
    unittest.main()
