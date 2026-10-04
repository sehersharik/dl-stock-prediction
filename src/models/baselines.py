from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import logging

logger = logging.getLogger(__name__)

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception as e:
    logger.warning(f"XGBoost could not be loaded: {e}")
    XGB_AVAILABLE = False

class MajorityClassBaseline:
    """Predicts whatever class was most frequent in the training set."""
    def fit(self, X, y):
        self.majority_class = int(np.bincount(y).argmax())
        return self
        
    def predict(self, X):
        return np.full(len(X), self.majority_class)
        
    def predict_proba(self, X):
        probs = np.zeros((len(X), 2))
        probs[:, self.majority_class] = 1.0
        return probs

def get_baseline_model(model_name: str):
    models = {
        "majority": MajorityClassBaseline(),
        "logistic": LogisticRegression(max_iter=2000, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    }
    
    if XGB_AVAILABLE:
        models["xgboost"] = XGBClassifier(eval_metric='logloss', random_state=42, n_jobs=-1)
        
    if model_name not in models:
        raise ValueError(f"Unknown baseline model or missing dependency: {model_name}")
    return models[model_name]
