import pandas as pd
import numpy as np
import torch
import joblib
import logging
from pathlib import Path
from src.features.generator import FeatureGenerator
from src.data.live_provider import LiveDataProvider
from src.models.dl_models import ConfigurableMLP
from sklearn.preprocessing import StandardScaler
from src.evaluation.splitter import TimeSeriesSplitter

logger = logging.getLogger(__name__)

# Market context tickers required by the stationary feature pipeline
CONTEXT_TICKERS = {
    "nifty": "^NSEI",
    "vix": "^INDIAVIX"
}

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
        models_dir = self.project_root / 'models'
        proc_dir = self.project_root / self.config['data'].get('processed_dir', 'data/processed')
        
        # Load chronological data to refit identical scaler (matches exact training preprocessing)
        X_full = pd.read_csv(proc_dir / f"{ticker}_X.csv", index_col=0, parse_dates=True)
        y_full = pd.read_csv(proc_dir / f"{ticker}_y.csv", index_col=0, parse_dates=True)['Target_Direction']
        
        self.expected_features = X_full.columns.tolist()
        
        splitter = TimeSeriesSplitter(self.config)
        (X_train, y_train), _, _ = splitter.split_train_val_test(X_full, y_full)
        self.scaler = StandardScaler()
        self.scaler.fit(X_train)
        
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
        model_path = models_dir / f"{ticker}_{model_name}_Target_Direction.pt"
        self.direction_model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.direction_model.eval()

    def _load_local_context(self) -> dict:
        """Fall back to locally-saved 10y CSVs for context features."""
        raw_dir = self.project_root / self.config['data'].get('raw_dir', 'data/raw')
        context = {}
        mappings = {"nifty": "NSEI_10y.csv", "vix": "INDIAVIX_10y.csv"}
        for key, fname in mappings.items():
            fpath = raw_dir / fname
            if fpath.exists():
                context[key] = pd.read_csv(fpath, index_col=0, parse_dates=True)
                logger.info(f"Loaded local context fallback: {fname} ({len(context[key])} rows)")
        return context

    def run_live_inference(self, ticker: str, model_name: str):
        """Executes a single live inference tick with strict validation."""
        self._load_artifacts(ticker, model_name)
        
        try:
            # 1. Fetch live data
            live_data = self.provider.get_latest_data(ticker)
            timestamp = live_data.pop('timestamp')
            status = live_data.pop('status')
            source = live_data.pop('source')
            _ = live_data.pop('interval')
            
            # Stale/Incomplete Data Check
            if status == 'Stale/Delayed':
                logger.warning('Data is stale/delayed. Proceeding with latest available trading session.')
                
            # 2. Update history buffer with latest row
            self.history_buffer.loc[timestamp] = [
                live_data['Open'], live_data['High'], live_data['Low'], 
                live_data['Close'], live_data['Volume']
            ]
            
            # 3. Generate stationary features with NIFTY/VIX context
            logging.getLogger('src.features.generator').setLevel(logging.WARNING)
            features_df = self.feature_gen.generate(self.history_buffer, context_dfs=self.context_buffers)
            latest_features = features_df.iloc[[-1]]
            
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
                "ticker": ticker,
                "model_version": self.model_version,
                "feature_version": self.feature_version,
                "direction_prob": dir_prob,
                "predicted_direction": direction,
                "risk_prob": None,
                "risk_category": None,
                "status": "Success",
                "latest_features": latest_features.iloc[-1].to_dict(),
                "price": live_data['Close'],
                "volume": live_data['Volume'],
                "open": live_data['Open'],
                "high": live_data['High'],
                "low": live_data['Low']
            }
            
        except Exception as e:
            logger.error(f"Live inference failed: {e}")
            return {"status": "Error", "message": str(e)}
