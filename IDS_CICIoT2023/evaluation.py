import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    precision_recall_curve, auc
)
from sklearn.preprocessing import label_binarize

def calculate_comprehensive_metrics(y_true, y_pred, y_prob=None, class_names=None):
    """
    Calculates comprehensive evaluation metrics for multi-class IDS:
    - Accuracy
    - Macro & Weighted Precision, Recall, F1-score
    - Specificity (TNR), False Positive Rate (FPR), False Negative Rate (FNR)
    - ROC-AUC and PR-AUC (if probabilities provided)
    - Per-class metrics
    """
    n_classes = len(np.unique(y_true)) if class_names is None else len(class_names)
    labels = np.arange(n_classes) if class_names is not None else np.unique(y_true)

    # Basic metrics
    acc = accuracy_score(y_true, y_pred)
    macro_prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    macro_rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    weighted_prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    weighted_rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # Compute confusion matrix per class for One-vs-Rest Specificity, FPR, FNR
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    specificities = []
    fprs = []
    fnrs = []

    total_samples = np.sum(cm)
    for i in range(len(labels)):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = total_samples - (tp + fn + fp)

        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        specificities.append(tnr)
        fprs.append(fpr)
        fnrs.append(fnr)

    macro_specificity = float(np.mean(specificities))
    macro_fpr = float(np.mean(fprs))
    macro_fnr = float(np.mean(fnrs))

    # ROC-AUC and PR-AUC
    roc_auc = np.nan
    pr_auc = np.nan

    if y_prob is not None:
        try:
            if y_prob.ndim == 2 and y_prob.shape[1] == n_classes:
                y_true_bin = label_binarize(y_true, classes=labels)
                if y_true_bin.shape[1] == 1 and n_classes == 2:
                    y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
                
                # ROC-AUC
                roc_auc = roc_auc_score(y_true_bin, y_prob, multi_class='ovr', average='macro')
                
                # PR-AUC
                pr_aucs = []
                for i in range(y_true_bin.shape[1]):
                    if np.sum(y_true_bin[:, i]) > 0:
                        p, r, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
                        pr_aucs.append(auc(r, p))
                if len(pr_aucs) > 0:
                    pr_auc = float(np.mean(pr_aucs))
        except Exception as e:
            print(f"[METRICS WARNING] Failed to calculate ROC/PR AUC: {e}")

    metrics = {
        'Accuracy': float(acc),
        'Precision': float(macro_prec),
        'Recall': float(macro_rec),
        'F1_Score': float(macro_f1),
        'Specificity': float(macro_specificity),
        'FPR': float(macro_fpr),
        'FNR': float(macro_fnr),
        'ROC_AUC': float(roc_auc) if not np.isnan(roc_auc) else 0.0,
        'PR_AUC': float(pr_auc) if not np.isnan(pr_auc) else 0.0,
        'Macro_Precision': float(macro_prec),
        'Macro_Recall': float(macro_rec),
        'Macro_F1': float(macro_f1),
        'Weighted_Precision': float(weighted_prec),
        'Weighted_Recall': float(weighted_rec),
        'Weighted_F1': float(weighted_f1)
    }
    return metrics, cm

def plot_and_save_confusion_matrix(cm, class_names, output_png_path, output_csv_path, model_name="Model"):
    """
    Plots and saves confusion matrix heatmap as PNG and numerical confusion matrix as CSV.
    """
    os.makedirs(os.path.dirname(output_png_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    # Save numerical confusion matrix CSV
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    df_cm.to_csv(output_csv_path)

    # Plot PNG heatmap
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title(f'Confusion Matrix — {model_name}', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Predicted Label', fontsize=12, labelpad=10)
    ax.set_ylabel('Actual Label', fontsize=12, labelpad=10)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_png_path, dpi=300)
    plt.close(fig)
    print(f"[EVALUATION] Saved confusion matrix PNG: {output_png_path}")
    print(f"[EVALUATION] Saved confusion matrix CSV: {output_csv_path}")

def plot_and_save_training_curves(history, accuracy_png_path, loss_png_path, model_name="Model"):
    """
    Plots and saves training & validation accuracy and loss curves for DL models.
    """
    os.makedirs(os.path.dirname(accuracy_png_path), exist_ok=True)
    os.makedirs(os.path.dirname(loss_png_path), exist_ok=True)

    hist_dict = history.history if hasattr(history, 'history') else history

    acc = hist_dict.get('accuracy', hist_dict.get('acc', []))
    val_acc = hist_dict.get('val_accuracy', hist_dict.get('val_acc', []))
    loss = hist_dict.get('loss', [])
    val_loss = hist_dict.get('val_loss', [])

    epochs = range(1, len(acc) + 1)

    # Accuracy Curve
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(epochs, acc, 'b-o', label='Training Accuracy', linewidth=2)
    if len(val_acc) > 0:
        ax.plot(epochs, val_acc, 'r-s', label='Validation Accuracy', linewidth=2)
    ax.set_title(f'Training & Validation Accuracy — {model_name}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epochs', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(accuracy_png_path, dpi=300)
    plt.close(fig)

    # Loss Curve
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(epochs, loss, 'b-o', label='Training Loss', linewidth=2)
    if len(val_loss) > 0:
        ax.plot(epochs, val_loss, 'r-s', label='Validation Loss', linewidth=2)
    ax.set_title(f'Training & Validation Loss — {model_name}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epochs', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(loss_png_path, dpi=300)
    plt.close(fig)

    print(f"[EVALUATION] Saved accuracy curve: {accuracy_png_path}")
    print(f"[EVALUATION] Saved loss curve: {loss_png_path}")
