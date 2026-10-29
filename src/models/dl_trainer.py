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
                       seq_length, scaler, model, history, metrics, test_trues, test_preds, test_probs, dl_cfg,
                       val_pred_df: pd.DataFrame = None):
        """Persist a run. test_trues/test_preds/test_probs may be None when the test set was
        deliberately not evaluated (e.g. during validation-only feature selection)."""
        # Paths
        ticker_clean = ticker.replace(".NS", "")
        registry_dir = self.project_root / 'models' / ticker_clean / model_name / run_id
        registry_dir.mkdir(parents=True, exist_ok=False)
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
            "training_period": [str(X_train.index[0]), str(X_train.index[-1])] if len(X_train.index) else [],
            "validation_period": [str(X_val.index[0]), str(X_val.index[-1])] if len(X_val.index) else [],
            "test_period": [str(X_test.index[0]), str(X_test.index[-1])] if len(X_test.index) else [],
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
        if val_pred_df is not None:
            val_pred_df.assign(Run_ID=run_id).to_csv(registry_dir / "val_preds.csv")
        if test_trues is None:
            logger.info(f"Model [{run_id}] saved to registry without test evaluation: {registry_dir}")
            return
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
        
    def _predict(self, model, loader):
        probs, trues = [], []
        with torch.no_grad():
            for batch_x, batch_y in loader:
                probs.extend(torch.sigmoid(model(batch_x.to(self.device))).cpu().numpy())
                trues.extend(batch_y.numpy())
        probs, trues = np.array(probs).flatten(), np.array(trues).flatten()
        return trues, (probs >= 0.5).astype(int), probs

    def train_mlp(self, ticker: str, target_col: str = 'Target_Direction', feature_cols: list = None,
                  evaluate_test: bool = True):
        """Train the MLP with early stopping on validation loss.

        Returns {"run_id", "val": metrics, "test": metrics or None}. With evaluate_test=False the
        test split is never scored, so the run can be used for model/feature selection.
        """
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
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
        layers = dl_cfg.get('layers', [32, 16])
        dropout = dl_cfg.get('dropout', 0.2)
        lr = dl_cfg.get('learning_rate', 0.001)
        weight_decay = dl_cfg.get('weight_decay', 1e-3)
        epochs = dl_cfg.get('epochs', 60)
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
        
        model = ConfigurableMLP(input_dim=X_train.shape[1], layer_dims=layers, dropout=dropout, use_batch_norm=False).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
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
        
        val_trues, val_preds, val_probs = self._predict(model, val_loader)
        metrics = {"val": evaluate_classification(val_trues, val_preds, val_probs), "test": None}
        val_pred_df = pd.DataFrame({'True': val_trues, 'Pred': val_preds, 'Prob': val_probs}, index=X_val.index)
        
        test_trues = test_preds = test_probs = None
        if evaluate_test:
            test_trues, test_preds, test_probs = self._predict(model, test_loader)
            metrics["test"] = evaluate_classification(test_trues, test_preds, test_probs)
        
        self._save_registry(run_id, "mlp", ticker, target_col, feature_cols, X_train, X_val, X_test, 1, scaler, model, history, metrics, test_trues, test_preds, test_probs, dl_cfg,
                            val_pred_df=val_pred_df)
        
        return {"run_id": run_id, **metrics}

    def _fit_binary(self, model, train_loader, val_loader, y_train: np.ndarray, dl_cfg: dict):
        """Class-weighted BCE training with early stopping on validation loss (validation only)."""
        num_pos = y_train.sum()
        pos_weight = torch.tensor([(len(y_train) - num_pos) / max(num_pos, 1)], dtype=torch.float32).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=dl_cfg.get('learning_rate', 0.001),
                                     weight_decay=dl_cfg.get('weight_decay', 1e-3))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        patience = dl_cfg.get('early_stopping_patience', 10)
        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        best_val_loss, best_state, no_improve = float('inf'), None, 0
        for epoch in range(dl_cfg.get('epochs', 60)):
            stats = {}
            for phase, loader in (("train", train_loader), ("val", val_loader)):
                model.train(phase == "train")
                loss_sum, correct, total = 0.0, 0, 0
                with torch.set_grad_enabled(phase == "train"):
                    for bx, by in loader:
                        bx, by = bx.to(self.device), by.to(self.device)
                        logits = model(bx)
                        loss = criterion(logits, by)
                        if phase == "train":
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                        loss_sum += loss.item() * bx.size(0)
                        correct += ((torch.sigmoid(logits) >= 0.5).float() == by).sum().item()
                        total += by.size(0)
                stats[phase] = (loss_sum / total, correct / total)
            for phase in ("train", "val"):
                history[f'{phase}_loss'].append(stats[phase][0])
                history[f'{phase}_acc'].append(stats[phase][1])
            v_l = stats["val"][0]
            scheduler.step(v_l)
            if v_l < best_val_loss:
                best_val_loss, best_state, no_improve = v_l, copy.deepcopy(model.state_dict()), 0
            else:
                no_improve += 1
            if no_improve >= patience:
                break
        model.load_state_dict(best_state)
        model.eval()
        return history, len(history['val_loss'])

    def train_sequence(self, ticker: str, cell: str = "lstm", seq_length: int = 20, target_col: str = 'Target_Direction',
                       feature_cols: list = None):
        """LSTM/GRU on windows of the last `seq_length` sessions, using the same fixed-date
        splits as the MLP. A window belongs to the split of its last day, so the test set is
        the same days as the MLP's; earlier days in a window are features only (no labels)."""
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        seed_everything(42)
        proc_dir = self.project_root / self.config['data'].get('processed_dir', 'data/processed')
        X = pd.read_csv(proc_dir / f"{ticker}_X.csv", index_col=0, parse_dates=True)
        if feature_cols is not None: X = X[feature_cols]
        else: feature_cols = X.columns.tolist()
        y = pd.read_csv(proc_dir / f"{ticker}_y.csv", index_col=0, parse_dates=True)[target_col]
        
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = self.splitter.split_train_val_test(X, y)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(X_train)  # fit on training rows only
        X_used, y_used = X.loc[:X_test.index[-1]], y.loc[:X_test.index[-1]]
        X_scaled = pd.DataFrame(scaler.transform(X_used), index=X_used.index, columns=X_used.columns)
        X_seq, y_seq, ends = self._create_sequences(X_scaled, y_used, seq_length)
        ends = pd.DatetimeIndex(ends)
        masks = {name: ends.isin(split.index) for name, split in (("train", X_train), ("val", X_val), ("test", X_test))}
        
        dl_cfg = self.config['models'].get('deep_learning', {})
        batch_size = dl_cfg.get('batch_size', 64)
        def make_loader(mask, shuffle=False):
            return DataLoader(TensorDataset(torch.FloatTensor(X_seq[mask]), torch.FloatTensor(y_seq[mask]).unsqueeze(1)),
                              batch_size=batch_size, shuffle=shuffle)
        loaders = {"train": make_loader(masks["train"], shuffle=True), "val": make_loader(masks["val"]), "test": make_loader(masks["test"])}
        
        model = ConfigurableLSTM(input_dim=len(feature_cols), lstm_dims=dl_cfg.get('lstm_dims', [32, 16]),
                                 dense_dims=dl_cfg.get('dense_dims', [16]), dropout=dl_cfg.get('dropout', 0.2), cell=cell).to(self.device)
        history, epochs_run = self._fit_binary(model, loaders["train"], loaders["val"], y_seq[masks["train"]], dl_cfg)
        
        val_trues, val_preds, val_probs = self._predict(model, loaders["val"])
        test_trues, test_preds, test_probs = self._predict(model, loaders["test"])
        metrics = {"val": evaluate_classification(val_trues, val_preds, val_probs),
                   "test": evaluate_classification(test_trues, test_preds, test_probs), "epochs_run": epochs_run}
        idx = {k: pd.DataFrame(index=ends[m]) for k, m in masks.items()}
        val_pred_df = pd.DataFrame({'True': val_trues, 'Pred': val_preds, 'Prob': val_probs}, index=ends[masks["val"]])
        self._save_registry(run_id, cell, ticker, target_col, feature_cols, idx["train"], idx["val"], idx["test"], seq_length,
                            scaler, model, history, metrics, test_trues, test_preds, test_probs, dl_cfg, val_pred_df=val_pred_df)
        return {"run_id": run_id, "cell": cell, "seq_length": seq_length, **metrics}

    def train_lstm(self, ticker: str, target_col: str = 'Target_Direction', feature_cols: list = None):
        """LSTM with the configured sequence length, on the same fixed-date splits as the MLP."""
        seq_length = self.config['features'].get('sequence_length', 30)
        return self.train_sequence(ticker, cell="lstm", seq_length=seq_length, target_col=target_col, feature_cols=feature_cols)
