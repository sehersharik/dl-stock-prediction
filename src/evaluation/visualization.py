import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, ConfusionMatrixDisplay
import os
import numpy as np

def plot_training_history(history: dict, save_path: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_title('Training vs Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Accuracy (if captured)
    if 'train_acc' in history and len(history['train_acc']) > 0:
        ax2.plot(history['train_acc'], label='Train Acc')
        ax2.plot(history['val_acc'], label='Val Acc')
        ax2.set_title('Training vs Validation Accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_roc_curve(y_true, y_prob, save_path: str):
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.figure(figsize=(6,6))
        plt.plot(fpr, tpr, color='blue', lw=2)
        plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        pass

def plot_pr_curve(y_true, y_prob, save_path: str):
    try:
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        plt.figure(figsize=(6,6))
        plt.plot(recall, precision, color='green', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        pass

def plot_confusion_matrix(cm, save_path: str):
    disp = ConfusionMatrixDisplay(confusion_matrix=np.array(cm))
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
