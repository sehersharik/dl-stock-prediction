import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score, 
                             recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix)

def evaluate_classification(y_true, y_pred, y_prob=None) -> dict:
    metrics = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Balanced Accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "Confusion Matrix": confusion_matrix(y_true, y_pred).tolist()
    }
    
    if y_prob is not None:
        try:
            metrics["ROC-AUC"] = float(roc_auc_score(y_true, y_prob))
            metrics["PR-AUC"] = float(average_precision_score(y_true, y_prob))
        except ValueError:
            metrics["ROC-AUC"] = None
            metrics["PR-AUC"] = None
            
    return metrics
