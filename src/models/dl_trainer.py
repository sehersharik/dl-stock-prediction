import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import copy
import logging
import json
import joblib
import subprocess
from datetime import datetime
from pathlib import Path

from src.models.dl_models import ConfigurableMLP, ConfigurableLSTM, seed_everything
from src.evaluation.splitter import TimeSeriesSplitter
from src.evaluation.metrics import evaluate_classification
from src.evaluation.plots import plot_training_history, plot_evaluation_curves

logger = logging.getLogger(__name__)

class DLTrainer:
    def __init__(self, config: dict, project_root: Path):
        self.config = config
        self.project_root = project_root
        self.splitter = TimeSeriesSplitter(config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        
    def _get_git_commit(self):
        try:
            return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=self.project_root, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return "unknown"
            
    def _save_registry(self, run_id, model_name, ticker, target_col, feature_cols, X_train, X_val, X_test, 
                       seq_length, scaler, model, history, metrics, test_trues, test_preds, test_probs, dl_cfg):
        
        # Paths
        ticker_clean = ticker.replace(".NS", "")
        registry_dir = self.project_root / 'models' / ticker_clean / model_name / run_id
        registry_dir.mkdir(parents=True, exist_ok=True)
        pred_dir = self.project_root / 'data' / 'predictions'
        pred_dir.mkdir(parents=True, exist_ok=True)
        
        # Save Model and Scaler natively grouped together
        model_path = registry_dir / "model.pt"
        scaler_path = registry_dir / "scaler.joblib"
        torch.save(model.state_dict(), model_path)
        joblib.dump(scaler, scaler_path)
        
        # Construct Metadata strictly
        metadata = {
            "run_id": run_id,
            "model_type": model_name,
            "ticker": ticker,
            "target": target_col,
            "feature_set": f"Count: {len(feature_cols)}",
            "feature_list": feature_cols,
            "training_date": datetime.now().isoformat(),
            "training_period": [str(X_train.index[0]), str(X_train.index[-1])] if not X_train.empty else [],
            "validation_period": [str(X_val.index[0]), str(X_val.index[-1])] if not X_val.empty else [],
            "test_period": [str(X_test.index[0]), str(X_test.index[-1])] if not X_test.empty else [],
            "sequence_length": seq_length,
            "scaler_path": str(scaler_path),
            "random_seed": 42,
            "hyperparameters": dl_cfg,
            "metrics": metrics,
            "git_commit": self._get_git_commit()
        }
        
        with open(registry_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)
            
        plot_training_history(history, registry_dir / "training_history.png")
        plot_evaluation_curves(test_trues, test_probs, test_preds, registry_dir, "evaluation")
        
        # Save Predictions with run_id explicitly referencing the authoring model
        df_pred = pd.DataFrame({
            'True': test_trues, 'Pred': test_preds, 'Prob': test_probs, 'Run_ID': run_id
        }, index=X_test.index[-len(test_trues):]) 
        
        df_pred.to_csv(registry_dir / "test_preds.csv")
        df_pred.to_csv(pred_dir / f"{ticker}_{model_name}_{target_col}_test_preds.csv")
        
        logger.info(f"Model [{run_id}] metadata completely saved to registry: {registry_dir}")
        
    def _create_sequences(self, X: pd.DataFrame, y: pd.Series, seq_length: int = 30):
        Xs, ys, indices = [], [], []
        for i in range(len(X) - seq_length + 1):
            Xs.append(X.iloc[i : i + seq_length].values)
            ys.append(y.iloc[i + seq_length - 1])
            indices.append(X.index[i + seq_length - 1])
        return np.array(Xs), np.array(ys), np.array(indices)
        
    def train_mlp(self, ticker: str, target_col: str = 'Target_Direction', feature_cols: list = None):
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        seed_everything(42)
        proc_dir = self.project_root / self.config['data'].get('processed_dir', 'data/processed')
        
        X = pd.read_csv(proc_dir / f"{ticker}_X.csv", index_col=0, parse_dates=True)
        if feature_cols is not None: X = X[feature_cols]
        else: feature_cols = X.columns.tolist()
        
        y = pd.read_csv(proc_dir / f"{ticker}_y.csv", index_col=0, parse_dates=True)[target_col]
        
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = self.splitter.split_train_val_test(X, y)
        scaler, X_train_s, X_val_s, X_test_s = self.splitter.scale_features(X_train, X_val, X_test)
        
        dl_cfg = self.config['models'].get('deep_learning', {})
        batch_size = dl_cfg.get('batch_size', 64)
        layers = dl_cfg.get('layers', [128, 64, 32])
        dropout = dl_cfg.get('dropout', 0.3)
        lr = dl_cfg.get('learning_rate', 0.001)
        epochs = dl_cfg.get('epochs', 100)
        patience = dl_cfg.get('early_stopping_patience', 10)
        
        def make_loader(X_df, y_df, shuffle=False):
            tX = torch.FloatTensor(X_df.values)
            ty = torch.FloatTensor(y_df.values).unsqueeze(1)
            return DataLoader(TensorDataset(tX, ty), batch_size=batch_size, shuffle=shuffle)
            
        train_loader = make_loader(X_train_s, y_train, shuffle=True)
        val_loader = make_loader(X_val_s, y_val)
        test_loader = make_loader(X_test_s, y_test)
        
        num_pos = y_train.sum()
        num_neg = len(y_train) - num_pos
        pos_weight = torch.tensor([num_neg / max(num_pos, 1)], dtype=torch.float32).to(self.device)
        
        model = ConfigurableMLP(input_dim=X_train.shape[1], layer_dims=layers, dropout=dropout).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        best_val_loss = float('inf')
        best_model_state = None
        epochs_no_improve = 0
        
        for epoch in range(epochs):
            model.train()
            tr_loss, tr_correct, tr_total = 0.0, 0, 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                tr_loss += loss.item() * batch_x.size(0)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                tr_correct += (preds == batch_y).sum().item()
                tr_total += batch_y.size(0)
                
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    logits = model(batch_x)
                    loss = criterion(logits, batch_y)
                    val_loss += loss.item() * batch_x.size(0)
                    preds = (torch.sigmoid(logits) >= 0.5).float()
                    val_correct += (preds == batch_y).sum().item()
                    val_total += batch_y.size(0)
                    
            t_l, v_l = tr_loss / tr_total, val_loss / val_total
            history['train_loss'].append(t_l)
            history['val_loss'].append(v_l)
            history['train_acc'].append(tr_correct / tr_total)
            history['val_acc'].append(val_correct / val_total)
            
            scheduler.step(v_l)
            if v_l < best_val_loss:
                best_val_loss = v_l
                best_model_state = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                
            if epochs_no_improve >= patience: break
                
        model.load_state_dict(best_model_state)
        model.eval()
        
        test_preds, test_probs, test_trues = [], [], []
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                logits = model(batch_x)
                probs = torch.sigmoid(logits).cpu().numpy()
                test_probs.extend(probs)
                test_preds.extend((probs >= 0.5).astype(int))
                test_trues.extend(batch_y.numpy())
                
        test_preds, test_probs, test_trues = np.array(test_preds).flatten(), np.array(test_probs).flatten(), np.array(test_trues).flatten()
        metrics = evaluate_classification(test_trues, test_preds, test_probs)
        
        self._save_registry(run_id, "mlp", ticker, target_col, feature_cols, X_train, X_val, X_test, 1, scaler, model, history, metrics, test_trues, test_preds, test_probs, dl_cfg)
        
        return metrics

    def train_lstm(self, ticker: str, target_col: str = 'Target_Direction', feature_cols: list = None):
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        seed_everything(42)
        proc_dir = self.project_root / self.config['data'].get('processed_dir', 'data/processed')
        
        X = pd.read_csv(proc_dir / f"{ticker}_X.csv", index_col=0, parse_dates=True)
        if feature_cols is not None: X = X[feature_cols]
        else: feature_cols = X.columns.tolist()
        y = pd.read_csv(proc_dir / f"{ticker}_y.csv", index_col=0, parse_dates=True)[target_col]
        
        n = len(X)
        train_end_idx = int(n * self.splitter.train_pct)
        val_end_idx = int(n * (self.splitter.train_pct + self.splitter.val_pct))
        
        train_cutoff, val_cutoff = X.index[train_end_idx - 1], X.index[val_end_idx - 1]
        
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(X.iloc[:train_end_idx])
        X_scaled = pd.DataFrame(scaler.transform(X), index=X.index, columns=X.columns)
        
        dl_cfg = self.config['models'].get('deep_learning', {})
        seq_length = dl_cfg.get('sequence_length', 30)
        X_seq, y_seq, seq_indices = self._create_sequences(X_scaled, y, seq_length)
        
        train_mask = seq_indices <= train_cutoff
        val_mask = (seq_indices > train_cutoff) & (seq_indices <= val_cutoff)
        test_mask = seq_indices > val_cutoff
        
        X_train, y_train = X_seq[train_mask], y_seq[train_mask]
        X_val, y_val = X_seq[val_mask], y_seq[val_mask]
        X_test, y_test = X_seq[test_mask], y_seq[test_mask]
        
        X_train_df = pd.DataFrame(index=seq_indices[train_mask])
        X_val_df = pd.DataFrame(index=seq_indices[val_mask])
        X_test_df = pd.DataFrame(index=seq_indices[test_mask])
        
        batch_size = dl_cfg.get('batch_size', 64)
        def make_loader(X_arr, y_arr, shuffle=False):
            return DataLoader(TensorDataset(torch.FloatTensor(X_arr), torch.FloatTensor(y_arr).unsqueeze(1)), batch_size=batch_size, shuffle=shuffle)
            
        train_loader = make_loader(X_train, y_train, shuffle=True)
        val_loader = make_loader(X_val, y_val)
        test_loader = make_loader(X_test, y_test)
        
        lstm_dims = dl_cfg.get('lstm_dims', [64, 32])
        dense_dims = dl_cfg.get('dense_dims', [16])
        dropout = dl_cfg.get('dropout', 0.3)
        lr = dl_cfg.get('learning_rate', 0.001)
        epochs = dl_cfg.get('epochs', 100)
        patience = dl_cfg.get('early_stopping_patience', 10)
        
        model = ConfigurableLSTM(input_dim=X_train.shape[2], lstm_dims=lstm_dims, dense_dims=dense_dims, dropout=dropout).to(self.device)
        
        num_pos = y_train.sum()
        num_neg = len(y_train) - num_pos
        pos_weight = torch.tensor([num_neg / max(num_pos, 1)], dtype=torch.float32).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        best_val_loss = float('inf')
        best_model_state = None
        epochs_no_improve = 0
        
        for epoch in range(epochs):
            model.train()
            tr_loss, tr_correct, tr_total = 0.0, 0, 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                tr_loss += loss.item() * batch_x.size(0)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                tr_correct += (preds == batch_y).sum().item()
                tr_total += batch_y.size(0)
                
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    logits = model(batch_x)
                    loss = criterion(logits, batch_y)
                    val_loss += loss.item() * batch_x.size(0)
                    preds = (torch.sigmoid(logits) >= 0.5).float()
                    val_correct += (preds == batch_y).sum().item()
                    val_total += batch_y.size(0)
            
            t_l, v_l = tr_loss / tr_total, val_loss / val_total
            history['train_loss'].append(t_l)
            history['val_loss'].append(v_l)
            history['train_acc'].append(tr_correct / tr_total)
            history['val_acc'].append(val_correct / val_total)
            
            scheduler.step(v_l)
            if v_l < best_val_loss:
                best_val_loss = v_l
                best_model_state = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                
            if epochs_no_improve >= patience: break
                
        model.load_state_dict(best_model_state)
        model.eval()
        
        test_preds, test_probs, test_trues = [], [], []
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                logits = model(batch_x)
                probs = torch.sigmoid(logits).cpu().numpy()
                test_probs.extend(probs)
                test_preds.extend((probs >= 0.5).astype(int))
                test_trues.extend(batch_y.numpy())
                
        test_preds, test_probs, test_trues = np.array(test_preds).flatten(), np.array(test_probs).flatten(), np.array(test_trues).flatten()
        metrics = evaluate_classification(test_trues, test_preds, test_probs)
        
        self._save_registry(run_id, "lstm", ticker, target_col, feature_cols, X_train_df, X_val_df, X_test_df, seq_length, scaler, model, history, metrics, test_trues, test_preds, test_probs, dl_cfg)
        
        return metrics
