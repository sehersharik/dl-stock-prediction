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

class InferenceEngine:
    def __init__(self, config: dict, provider: LiveDataProvider, project_root: Path):
        self.config = config
        self.provider = provider
        self.project_root = project_root
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        
        self.feature_gen = FeatureGenerator(self.config)
        self.model_version = "v1.0.0"
        self.feature_version = "v1.0.0"
        
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
        
        # Load rolling buffer for live feature calculations using live provider
        try:
            self.history_buffer = self.provider.get_lookback_data(ticker, lookback_days=250)
        except Exception as e:
            logger.error(f"Failed to fetch live lookback data. Falling back to local CSV: {e}")
            raw_dir = self.project_root / self.config['data'].get('raw_dir', 'data/raw')
            files = list(raw_dir.glob(f"{ticker}_*.csv"))
            if not files:
                raise FileNotFoundError("Raw historical data needed for feature warm-up.")
            latest_file = sorted(files)[-1]
            self.history_buffer = pd.read_csv(latest_file, index_col=0, parse_dates=True).tail(250)
        
        # Load Direction Model
        dl_cfg = self.config['models'].get('deep_learning', {})
        layers = dl_cfg.get('layers', [128, 64, 32])
        dropout = dl_cfg.get('dropout', 0.3)
        
        self.direction_model = ConfigurableMLP(input_dim=len(self.expected_features), layer_dims=layers, dropout=dropout).to(self.device)
        model_path = models_dir / f"{ticker}_{model_name}_Target_Direction.pt"
        self.direction_model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.direction_model.eval()

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
                
            # 2. Update Feature Calculations
            self.history_buffer.loc[timestamp] = [
                live_data['Open'], live_data['High'], live_data['Low'], 
                live_data['Close'], live_data['Volume']
            ]
            self.history_buffer = self.history_buffer.tail(250)
            
            logging.getLogger('src.features.generator').setLevel(logging.WARNING)
            features_df = self.feature_gen.generate(self.history_buffer, context_dfs={})
            latest_features = features_df.iloc[[-1]]
            
            # Validation: Required features exist & Correct Order
            missing_cols = set(self.expected_features) - set(latest_features.columns)
            if missing_cols:
                logger.error(f"Prediction unavailable — missing features: {missing_cols}")
                return {"status": "Error", "message": f"Prediction unavailable: live feature schema does not match the trained model. Missing: {missing_cols}"}
                
            latest_features = latest_features[self.expected_features]
            
            # Validation: No NaNs
            if latest_features.isnull().values.any():
                logger.error("Prediction unavailable — NaN values detected in live features.")
                return {"status": "Error", "message": "Prediction unavailable: NaN values detected in live features."}
                
            # 3 & 4. Construct input & Apply Exact Preprocessing
            scaled_features = self.scaler.transform(latest_features)
            
            # Validation: Correct Model Input Shape
            if scaled_features.shape[1] != self.scaler.n_features_in_:
                logger.error("Prediction unavailable — incorrect feature shape.")
                return {"status": "Error", "message": "Prediction unavailable: incorrect feature shape."}
                
            # 5. Load and Produce Prediction
            x_tensor = torch.FloatTensor(scaled_features).to(self.device)
            with torch.no_grad():
                dir_logits = self.direction_model(x_tensor)
                dir_prob = torch.sigmoid(dir_logits).item()
                
            direction = "UP" if dir_prob >= 0.5 else "DOWN/CASH"
            
            # 6. Logging strict output (No credentials)
            logger.info(f"--- LIVE PREDICTION ---")
            logger.info(f"Timestamp: {timestamp}")
            logger.info(f"Ticker: {ticker}")
            logger.info(f"Model Version: {self.model_version}")
            logger.info(f"Feature Version: {self.feature_version}")
            logger.info(f"Direction Probability: {dir_prob:.4f}")
            logger.info(f"Predicted Direction: {direction}")
            logger.info(f"Risk Probability: N/A (Model not deployed)")
            logger.info(f"Risk Category: N/A (Model not deployed)")
            
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

