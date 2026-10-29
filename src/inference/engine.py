import pandas as pd
import numpy as np
import torch
import joblib
import logging
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo
from src.features.generator import FeatureGenerator
from src.data.live_provider import LiveDataProvider
from src.models.dl_models import ConfigurableMLP
from src.evaluation.metrics_registry import active_run, load_metrics

logger = logging.getLogger(__name__)

# Market context tickers required by the stationary feature pipeline
CONTEXT_TICKERS = {
    "NIFTY": "^NSEI",
    "VIX": "^INDIAVIX"
}

# NSE closes at 15:30 IST. A daily bar dated today is only treated as a completed
# session after this time, leaving a margin for the provider to settle the close.
SESSION_COMPLETE_IST = time(16, 0)


def to_daily_index(df: pd.DataFrame) -> pd.DataFrame:
    """Index by calendar date (one row per session), matching the daily bars used in training."""
    df = df.copy()
    df.index = pd.DatetimeIndex(df.index).normalize()
    return df[~df.index.duplicated(keep='last')].sort_index()


def drop_incomplete_session(df: pd.DataFrame, now: datetime) -> pd.DataFrame:
    """Drop today's bar if the trading session has not finished yet (`now` is naive IST)."""
    if len(df) and df.index[-1] == pd.Timestamp(now.date()) and now.time() < SESSION_COMPLETE_IST:
        return df.iloc[:-1]
    return df

class InferenceEngine:
    def __init__(self, config: dict, provider: LiveDataProvider, project_root: Path):
        self.config = config
        self.provider = provider
        self.project_root = project_root
        self.device = torch.device(
            'cuda' if torch.cuda.is_available()
            else 'mps' if torch.backends.mps.is_available()
            else 'cpu'
        )
        
        self.feature_gen = FeatureGenerator(self.config)
        self.model_version = "v2.0.0"   # 10y stationary pipeline
        self.feature_version = "v2.0.0"
        
    def _load_artifacts(self, ticker: str, model_name: str):
        # The active run in reports/metrics.json names the model, its fitted scaler and its
        # feature list, so live predictions always use the run whose metrics are displayed.
        run = active_run(load_metrics(self.project_root / "reports" / "metrics.json"))
        if run["model"]["type"] != model_name:
            raise ValueError(f"Active run {run['run_id']} is a {run['model']['type']} model, not {model_name}")
        self.run_id = run["run_id"]
        self.expected_features = run["model"]["feature_list"]
        self.scaler = joblib.load(self.project_root / run["artifacts"]["scaler"])
        
        # Fetch live lookback data for the target ticker (needs 2y for SMA_200 warmup)
        try:
            self.history_buffer = self.provider.get_lookback_data(ticker, lookback_days=260)
            logger.info(f"Fetched live lookback for {ticker}: {len(self.history_buffer)} rows")
        except Exception as e:
            logger.error(f"Failed to fetch live lookback data. Falling back to local CSV: {e}")
            raw_dir = self.project_root / self.config['data'].get('raw_dir', 'data/raw')
            # Prefer 10y file, fall back to any available file
            preferred = raw_dir / f"{ticker}_10y.csv"
            if preferred.exists():
                self.history_buffer = pd.read_csv(preferred, index_col=0, parse_dates=True)
            else:
                files = sorted(raw_dir.glob(f"{ticker}_*.csv"))
                if not files:
                    raise FileNotFoundError("Raw historical data needed for feature warm-up.")
                self.history_buffer = pd.read_csv(files[-1], index_col=0, parse_dates=True)
        
        # Fetch live NIFTY and VIX context data
        try:
            self.context_buffers = self.provider.get_context_data(CONTEXT_TICKERS, lookback_days=260)
            logger.info(f"Fetched context data for: {list(self.context_buffers.keys())}")
        except Exception as e:
            logger.warning(f"Failed to fetch live context data. Falling back to local CSVs: {e}")
            self.context_buffers = self._load_local_context()
        
        if not self.context_buffers:
            logger.warning("Context data unavailable — NIFTY/VIX features will be NaN in live inference.")
        
        # Load Direction Model — use config-driven layers matching training
        dl_cfg = self.config['models'].get('deep_learning', {})
        layers = dl_cfg.get('layers', [32, 16])
        dropout = dl_cfg.get('dropout', 0.2)
        
        self.direction_model = ConfigurableMLP(
            input_dim=len(self.expected_features),
            layer_dims=layers,
            dropout=dropout
        ).to(self.device)
        model_path = self.project_root / run["artifacts"]["model"]
        self.direction_model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.direction_model.eval()

    def _load_local_context(self) -> dict:
        """Fall back to locally-saved 10y CSVs for context features."""
        raw_dir = self.project_root / self.config['data'].get('raw_dir', 'data/raw')
        context = {}
        mappings = {"NIFTY": "NSEI_10y.csv", "VIX": "INDIAVIX_10y.csv"}
        for key, fname in mappings.items():
            fpath = raw_dir / fname
            if fpath.exists():
                context[key] = pd.read_csv(fpath, index_col=0, parse_dates=True)
                logger.info(f"Loaded local context fallback: {fname} ({len(context[key])} rows)")
        return context

    def run_live_inference(self, ticker: str, model_name: str, now: datetime = None):
        """Executes a single live inference tick with strict validation.

        Features are built from the latest *completed* daily session, as in training.
        The intraday tick is only used for the displayed price.
        """
        self._load_artifacts(ticker, model_name)
        if now is None:
            now = datetime.now(ZoneInfo('Asia/Kolkata')).replace(tzinfo=None)

        try:
            # 1. Fetch live data (display only — a 1-minute bar is not a daily feature row)
            live_data = self.provider.get_latest_data(ticker)
            timestamp = live_data.pop('timestamp')
            status = live_data.pop('status')
            source = live_data.pop('source')
            _ = live_data.pop('interval')

            # Stale/Incomplete Data Check
            if status == 'Stale/Delayed':
                logger.warning('Data is stale/delayed. Proceeding with latest available trading session.')

            # 2. Align target and context data to one row per completed trading session
            missing_ctx = [k for k in CONTEXT_TICKERS
                           if self.context_buffers.get(k) is None or self.context_buffers[k].empty]
            if missing_ctx:
                logger.error(f"Prediction unavailable — no market context data for: {missing_ctx}")
                return {
                    "status": "Error",
                    "message": f"Prediction unavailable: market context data could not be retrieved for {', '.join(missing_ctx)}"
                }
            history = drop_incomplete_session(to_daily_index(self.history_buffer), now)
            context = {k: to_daily_index(v) for k, v in self.context_buffers.items()}
            session_date = history.index[-1]

            # 3. Generate stationary features with NIFTY/VIX context
            logging.getLogger('src.features.generator').setLevel(logging.WARNING)
            features_df = self.feature_gen.generate(history, context_dfs=context)

            # Validation: the feature row must belong to the latest completed session,
            # otherwise an older row would silently be presented as current.
            if features_df.empty or features_df.index[-1] != session_date:
                logger.error(f"Prediction unavailable — no complete feature row for session {session_date.date()}")
                return {
                    "status": "Error",
                    "message": f"Prediction unavailable: features could not be computed for the latest completed session ({session_date.date()}); NIFTY or VIX data for that date may be missing"
                }
            latest_features = features_df.iloc[[-1]]
            
            # Reference for the displayed price change: last completed session before the tick's date
            tick_day = pd.Timestamp(timestamp).normalize()
            before_tick = history[history.index < tick_day]
            reference_date = before_tick.index[-1] if len(before_tick) else None
            reference_close = float(before_tick['Close'].iloc[-1]) if len(before_tick) else None
            
            # Validation: Required features exist & Correct Order
            missing_cols = set(self.expected_features) - set(latest_features.columns)
            if missing_cols:
                logger.error(f"Prediction unavailable — missing features: {missing_cols}")
                return {
                    "status": "Error",
                    "message": f"Prediction unavailable: live feature schema does not match the trained model. Missing: {missing_cols}"
                }
                
            latest_features = latest_features[self.expected_features]
            
            # Validation: No NaNs
            if latest_features.isnull().values.any():
                nan_cols = latest_features.columns[latest_features.isnull().any()].tolist()
                logger.error(f"Prediction unavailable — NaN values in: {nan_cols}")
                return {
                    "status": "Error",
                    "message": f"Prediction unavailable: NaN values detected in features: {nan_cols}"
                }
                
            # 4. Apply exact preprocessing (same scaler fitted on training split)
            scaled_features = self.scaler.transform(latest_features)
            
            # Validation: Correct Model Input Shape
            if scaled_features.shape[1] != self.scaler.n_features_in_:
                logger.error("Prediction unavailable — incorrect feature shape.")
                return {"status": "Error", "message": "Prediction unavailable: incorrect feature shape."}
                
            # 5. Produce Prediction
            x_tensor = torch.FloatTensor(scaled_features).to(self.device)
            with torch.no_grad():
                dir_logits = self.direction_model(x_tensor)
                dir_prob = torch.sigmoid(dir_logits).item()
                
            direction = "UP" if dir_prob >= 0.5 else "DOWN/CASH"
            
            # 6. Structured logging (no credentials)
            logger.info(f"--- LIVE PREDICTION ---")
            logger.info(f"Timestamp: {timestamp}")
            logger.info(f"Ticker: {ticker}")
            logger.info(f"Model Version: {self.model_version}")
            logger.info(f"Feature Version: {self.feature_version}")
            logger.info(f"Direction Probability: {dir_prob:.4f}")
            logger.info(f"Predicted Direction: {direction}")
            
            return {
                "timestamp": timestamp,
                "feature_date": session_date,
                "ticker": ticker,
                "model_version": self.model_version,
                "feature_version": self.feature_version,
                "run_id": self.run_id,
                "direction_prob": dir_prob,
                "predicted_direction": direction,
                "risk_prob": None,
                "risk_category": None,
                "status": "Success",
                "latest_features": latest_features.iloc[-1].to_dict(),
                "price": live_data['Close'],
                "reference_date": reference_date,
                "reference_close": reference_close,
                "price_change": (live_data['Close'] / reference_close - 1) if reference_close else None,
                "volume": live_data['Volume'],
                "open": live_data['Open'],
                "high": live_data['High'],
                "low": live_data['Low']
            }
            
        except Exception as e:
            logger.error(f"Live inference failed: {e}")
            return {"status": "Error", "message": str(e)}
