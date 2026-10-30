import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    precision_recall_curve, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.base import clone

def calculate_multiclass_roc_auc(y_true, y_prob, class_names=None):
    """
    Computes a robust, mathematically sound multiclass ROC-AUC using One-vs-Rest (OvR).
    
    Parameters:
    -----------
    y_true : array-like of shape (n_samples,)
        Ground truth integer class indices (0 to n_classes-1).
    y_prob : array-like of shape (n_samples, n_classes)
        Complete predicted probability matrix from the model.
    class_names : list of str, optional
        Human-readable class names matching output indices.
        
    Returns:
    --------
    macro_roc_auc : float or np.nan
        Macro-averaged One-vs-Rest ROC-AUC across all evaluatable classes.
    per_class_rocs : dict
        Mapping of class_index -> roc_auc_score for each evaluatable class.
    omitted_classes : list of tuple (class_name, reason)
        Classes omitted due to lack of positive/negative samples in y_true.
    weighted_roc_auc : float or np.nan
        Sample-weighted One-vs-Rest ROC-AUC across evaluatable classes.
    """
    if y_prob is None:
        print("[ROC-AUC INFO] Predicted probabilities (y_prob) not provided. ROC-AUC is unavailable.")
        return np.nan, {}, [], np.nan

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=np.float64)

    # 1. Sample count alignment check
    if len(y_true) != len(y_prob):
        err_msg = (
            f"[ROC-AUC ERROR] Sample count mismatch: y_true has {len(y_true)} samples, "
            f"but y_prob has {len(y_prob)} samples."
        )
        print(err_msg)
        return np.nan, {}, [("ALL", err_msg)], np.nan

    n_samples = len(y_true)
    if n_samples == 0:
        print("[ROC-AUC ERROR] Empty test set provided (n_samples == 0).")
        return np.nan, {}, [("ALL", "Empty test set")], np.nan

    # 2. Dimensionality check
    n_classes = len(class_names) if class_names is not None else y_prob.shape[1] if y_prob.ndim == 2 else 2
    labels = np.arange(n_classes)

    # Handle 1D binary probabilities if needed
    if y_prob.ndim == 1:
        if n_classes == 2:
            y_prob = np.column_stack([1.0 - y_prob, y_prob])
        else:
            err_msg = f"[ROC-AUC ERROR] y_prob is 1D array but n_classes is {n_classes}. Expected 2D probability matrix."
            print(err_msg)
            return np.nan, {}, [("ALL", err_msg)], np.nan

    if y_prob.ndim != 2:
        err_msg = f"[ROC-AUC ERROR] y_prob must be 2D array, got shape {y_prob.shape}."
        print(err_msg)
        return np.nan, {}, [("ALL", err_msg)], np.nan

    if y_prob.shape[1] != n_classes:
        err_msg = (
            f"[ROC-AUC ERROR] Probability matrix column count ({y_prob.shape[1]}) "
            f"does not match expected number of classes ({n_classes})."
        )
        print(err_msg)
        return np.nan, {}, [("ALL", err_msg)], np.nan

    # 3. Numerical validity checks
    if np.isnan(y_prob).any():
        err_msg = "[ROC-AUC ERROR] y_prob contains NaN values. Calculation aborted."
        print(err_msg)
        return np.nan, {}, [("ALL", err_msg)], np.nan

    if np.isinf(y_prob).any():
        err_msg = "[ROC-AUC ERROR] y_prob contains infinite values. Calculation aborted."
        print(err_msg)
        return np.nan, {}, [("ALL", err_msg)], np.nan

    if (y_prob < -1e-7).any() or (y_prob > 1.0 + 1e-4).any():
        print("[ROC-AUC WARNING] Probabilities contain values outside [0, 1]. Clipping to valid range [0, 1].")
        y_prob = np.clip(y_prob, 0.0, 1.0)

    # 4. Check row sum normalization (each row must sum to ~1.0)
    row_sums = np.sum(y_prob, axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-2):
        print("[ROC-AUC WARNING] Some probability rows do not sum to 1.0. Re-normalizing rows.")
        safe_sums = np.where(row_sums == 0, 1.0, row_sums)
        y_prob = y_prob / safe_sums[:, np.newaxis]

    # 5. Label binarization
    y_true_bin = label_binarize(y_true, classes=labels)
    if y_true_bin.shape[1] == 1 and n_classes == 2:
        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])

    # 6. Per-class One-vs-Rest calculation with missing-class handling
    per_class_rocs = {}
    omitted_classes = []
    class_weights = []

    for c in range(n_classes):
        c_name = class_names[c] if class_names is not None and c < len(class_names) else f"Class_{c}"
        pos_count = int(np.sum(y_true_bin[:, c]))
        neg_count = n_samples - pos_count

        if pos_count == 0:
            omitted_classes.append((c_name, "0 positive test samples in ground truth"))
            continue
        if neg_count == 0:
            omitted_classes.append((c_name, "0 negative test samples in ground truth (all samples are positive)"))
            continue

        try:
            auc_val = roc_auc_score(y_true_bin[:, c], y_prob[:, c])
            per_class_rocs[c] = float(auc_val)
            class_weights.append(pos_count)
        except Exception as e:
            err_reason = f"Calculation failed: {str(e)}"
            omitted_classes.append((c_name, err_reason))
            print(f"[ROC-AUC WARNING] Class {c_name} (index {c}): {err_reason}")

    if omitted_classes:
        print(f"[ROC-AUC INFO] {len(omitted_classes)}/{n_classes} classes omitted from OvR ROC-AUC:")
        for name, reason in omitted_classes:
            print(f"  - {name}: {reason}")

    # 7. Compute final macro & weighted averages
    if len(per_class_rocs) == 0:
        print("[ROC-AUC ERROR] No classes had both positive and negative samples. ROC-AUC is undefined (returning NaN).")
        return np.nan, {}, omitted_classes, np.nan

    macro_roc_auc = float(np.mean(list(per_class_rocs.values())))
    weighted_roc_auc = float(np.average(list(per_class_rocs.values()), weights=class_weights))

    print(
        f"[ROC-AUC SUCCESS] Macro OvR ROC-AUC: {macro_roc_auc:.4f} "
        f"across {len(per_class_rocs)}/{n_classes} evaluatable classes "
        f"(Weighted: {weighted_roc_auc:.4f})"
    )

    return macro_roc_auc, per_class_rocs, omitted_classes, weighted_roc_auc

def calculate_comprehensive_metrics(y_true, y_pred, y_prob=None, class_names=None):
    """
    Calculates comprehensive evaluation metrics for multi-class IDS:
    - Accuracy
    - Macro & Weighted Precision, Recall, F1-score
    - Specificity (TNR), False Positive Rate (FPR), False Negative Rate (FNR)
    - Multiclass Macro One-vs-Rest ROC-AUC and PR-AUC
    - Returns metrics dict and confusion matrix.
    """
    n_classes = len(np.unique(y_true)) if class_names is None else len(class_names)
    labels = np.arange(n_classes) if class_names is not None else np.unique(y_true)

    # Basic classification metrics
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

    # Multiclass ROC-AUC Calculation
    roc_auc = np.nan
    pr_auc = np.nan

    if y_prob is not None:
        try:
            macro_auc, per_class_rocs, omitted_classes, _ = calculate_multiclass_roc_auc(
                y_true=y_true,
                y_prob=y_prob,
                class_names=class_names
            )
            roc_auc = macro_auc
        except Exception as e:
            print(f"[METRICS ERROR] Exception during ROC-AUC calculation: {type(e).__name__}: {e}")
            roc_auc = np.nan

        # PR-AUC Calculation
        try:
            y_true_bin = label_binarize(y_true, classes=labels)
            if y_true_bin.shape[1] == 1 and n_classes == 2:
                y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])

            pr_aucs = []
            max_c = min(y_true_bin.shape[1], y_prob.shape[1])
            for i in range(max_c):
                if np.sum(y_true_bin[:, i]) > 0:
                    p, r, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
                    pr_aucs.append(auc(r, p))
            if len(pr_aucs) > 0:
                pr_auc = float(np.mean(pr_aucs))
        except Exception as e:
            print(f"[METRICS ERROR] Exception during PR-AUC calculation: {type(e).__name__}: {e}")
            pr_auc = np.nan

    # Assemble comprehensive metrics
    # Important: Do NOT convert undefined or failed ROC_AUC / PR_AUC to 0.0! Preserve np.nan.
    metrics = {
        'Accuracy': float(acc),
        'Precision': float(macro_prec),
        'Recall': float(macro_rec),
        'F1_Score': float(macro_f1),
        'Specificity': float(macro_specificity),
        'FPR': float(macro_fpr),
        'FNR': float(macro_fnr),
        'ROC_AUC': float(roc_auc) if (roc_auc is not None and not np.isnan(roc_auc)) else np.nan,
        'PR_AUC': float(pr_auc) if (pr_auc is not None and not np.isnan(pr_auc)) else np.nan,
        'Macro_Precision': float(macro_prec),
        'Macro_Recall': float(macro_rec),
        'Macro_F1': float(macro_f1),
        'Weighted_Precision': float(weighted_prec),
        'Weighted_Recall': float(weighted_rec),
        'Weighted_F1': float(weighted_f1)
    }
    return metrics, cm

def plot_and_save_roc_curves(y_true, y_prob, class_names, output_png_path, model_name="Model", max_display_classes=8):
    """
    Computes and plots per-class One-vs-Rest ROC curves alongside the macro-average ROC curve.
    Saves a high-resolution (300 DPI) publication-quality PNG.
    """
    os.makedirs(os.path.dirname(output_png_path), exist_ok=True)
    n_classes = len(class_names)
    labels = np.arange(n_classes)

    y_true_bin = label_binarize(y_true, classes=labels)
    if y_true_bin.shape[1] == 1 and n_classes == 2:
        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])

    fpr_dict = {}
    tpr_dict = {}
    roc_auc_dict = {}

    valid_classes = []
    for i in range(min(n_classes, y_prob.shape[1])):
        if np.sum(y_true_bin[:, i]) > 0 and np.sum(y_true_bin[:, i]) < len(y_true):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            fpr_dict[i] = fpr
            tpr_dict[i] = tpr
            roc_auc_dict[i] = auc(fpr, tpr)
            valid_classes.append(i)

    if not valid_classes:
        print(f"[EVALUATION WARNING] No valid classes with binary variance for ROC curve in {model_name}.")
        return

    # Compute macro-average ROC curve across valid classes
    all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in valid_classes]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in valid_classes:
        mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])
    mean_tpr /= len(valid_classes)
    macro_auc_val = auc(all_fpr, mean_tpr)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot prominent Macro-Average ROC Curve
    ax.plot(
        all_fpr, mean_tpr,
        label=f'Macro-average ROC (AUC = {macro_auc_val:.4f})',
        color='navy', linestyle='-', linewidth=3.0
    )

    # Plot top N most frequent or representative valid classes
    display_indices = valid_classes[:max_display_classes]
    colors = plt.cm.tab10(np.linspace(0, 1, len(display_indices)))

    for idx, c in enumerate(display_indices):
        c_label = class_names[c] if c < len(class_names) else f"Class {c}"
        ax.plot(
            fpr_dict[c], tpr_dict[c],
            label=f'{c_label} (AUC = {roc_auc_dict[c]:.3f})',
            color=colors[idx], linestyle='--', linewidth=1.5, alpha=0.8
        )

    # Plot chance baseline
    ax.plot([0, 1], [0, 1], 'k:', linewidth=1.5, label='Random Chance (AUC = 0.5000)')

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel('False Positive Rate (FPR)', fontsize=12, labelpad=8)
    ax.set_ylabel('True Positive Rate (TPR / Recall)', fontsize=12, labelpad=8)
    ax.set_title(f'One-vs-Rest ROC Curves — {model_name} (Macro AUC = {macro_auc_val:.4f})', fontsize=14, fontweight='bold', pad=12)
    ax.legend(loc='lower right', fontsize=9, frameon=True)
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_png_path, dpi=300)
    plt.close(fig)
    print(f"[EVALUATION] Saved ROC curves plot: {output_png_path}")

def save_complete_predictions(y_true, y_pred, y_prob, class_names, output_csv_path, model_name="Model"):
    """
    Saves the complete prediction results including:
    - Actual_Label (human-readable)
    - Predicted_Label (human-readable)
    - Prediction_Confidence (max predicted probability)
    - Full probability distribution: one column per class ('prob_{class_name}').
    
    This preserves the complete probability matrix so ROC-AUC, PR-AUC, and calibrated
    decision thresholds can be recalculated at any time without retraining.
    """
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    n_samples = len(y_true)
    n_classes = len(class_names)

    actual_labels = [class_names[i] if i < n_classes else str(i) for i in y_true]
    pred_labels = [class_names[i] if i < n_classes else str(i) for i in y_pred]
    max_probs = np.max(y_prob, axis=1)

    data_dict = {
        'Actual_Label': actual_labels,
        'Predicted_Label': pred_labels,
        'Prediction_Confidence': max_probs
    }

    # Add individual probability columns for all classes
    for c_idx, c_name in enumerate(class_names):
        col_name = f"prob_{c_name}"
        if c_idx < y_prob.shape[1]:
            data_dict[col_name] = y_prob[:, c_idx]
        else:
            data_dict[col_name] = np.zeros(n_samples, dtype=np.float64)

    df_preds = pd.DataFrame(data_dict)
    df_preds.to_csv(output_csv_path, index=False)
    print(f"[{model_name.upper()} PREDICTIONS] Saved complete prediction probabilities ({df_preds.shape}) to {output_csv_path}")
    return df_preds

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

def perform_5_fold_cv(
    model_or_builder,
    X,
    y,
    is_keras=False,
    model_kwargs=None,
    fit_kwargs=None,
    n_splits=5,
    random_state=42,
    max_samples=30000,
    model_name="Model"
):
    """
    Performs 5-Fold Stratified Cross-Validation to evaluate generalization accuracy.
    Handles both scikit-learn estimators and Keras model factory callables.
    
    Parameters:
    -----------
    model_or_builder : sklearn estimator or callable
        Trained or unfitted estimator (cloned per fold) or Keras model creation function.
    X : np.ndarray
        Training features or sequence tensors (2D or 3D).
    y : np.ndarray
        Training ground truth class labels (1D array of integers).
    is_keras : bool, default=False
        Whether model_or_builder is a Keras model factory.
    model_kwargs : dict, optional
        Keyword arguments passed to model_or_builder if is_keras is True.
    fit_kwargs : dict, optional
        Keyword arguments passed to model.fit (e.g. epochs, batch_size, class_weight).
    n_splits : int, default=5
        Number of cross-validation folds.
    random_state : int, default=42
        Seed for stratified fold splitting.
    max_samples : int, default=30000
        Maximum sample count used for CV to maintain high throughput and avoid excessive runtimes.
        If len(X) <= max_samples, full data is used.
    model_name : str, default="Model"
        Name of model for progress reporting.
        
    Returns:
    --------
    cv_mean : float
        Mean validation accuracy across the 5 folds.
    cv_std : float
        Standard deviation of validation accuracy across the 5 folds.
    cv_scores : list of float
        Individual fold validation accuracies.
    """
    X = np.asarray(X)
    y = np.asarray(y)

    # Subsample if dataset is exceptionally large for fast, responsive execution
    if max_samples is not None and len(X) > max_samples:
        try:
            sss = StratifiedShuffleSplit(n_splits=1, train_size=max_samples, random_state=random_state)
            sub_idx, _ = next(sss.split(X, y))
        except ValueError:
            rng = np.random.RandomState(random_state)
            sub_idx = rng.choice(len(X), size=max_samples, replace=False)
        X_cv = X[sub_idx]
        y_cv = y[sub_idx]
        print(f"[{model_name} 5-FOLD CV] Subsampled {len(X):,} -> {len(X_cv):,} records for 5-fold cross-validation...")
    else:
        X_cv = X
        y_cv = y
        print(f"[{model_name} 5-FOLD CV] Using {len(X_cv):,} records for 5-fold cross-validation...")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_cv, y_cv)):
        X_fold_train, y_fold_train = X_cv[train_idx], y_cv[train_idx]
        X_fold_val, y_fold_val = X_cv[val_idx], y_cv[val_idx]

        if is_keras:
            import tensorflow as tf
            tf.keras.backend.clear_session()
            fold_model = model_or_builder(**(model_kwargs or {}))
            
            fold_fit_params = dict(fit_kwargs or {})
            epochs = fold_fit_params.pop('epochs', 5)
            batch_size = fold_fit_params.pop('batch_size', 128)
            class_weight = fold_fit_params.pop('class_weight', None)
            
            early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
            callbacks = [early_stop]
            
            fold_model.fit(
                X_fold_train, y_fold_train,
                validation_data=(X_fold_val, y_fold_val),
                epochs=epochs,
                batch_size=batch_size,
                class_weight=class_weight,
                callbacks=callbacks,
                verbose=0
            )
            val_probs = fold_model.predict(X_fold_val, batch_size=256, verbose=0)
            val_preds = np.argmax(val_probs, axis=1)
        else:
            fold_model = clone(model_or_builder)
            fold_model.fit(X_fold_train, y_fold_train)
            val_preds = fold_model.predict(X_fold_val)

        acc = float(accuracy_score(y_fold_val, val_preds))
        cv_scores.append(acc)
        print(f"  [{model_name} 5-FOLD CV] Fold {fold_idx + 1}/{n_splits} Validation Accuracy: {acc:.4f}")

    cv_mean = float(np.mean(cv_scores))
    cv_std = float(np.std(cv_scores))
    print(f"[{model_name} 5-FOLD CV RESULT] Mean Validation Accuracy: {cv_mean:.4f} (+/- {cv_std:.4f})\n")
    return cv_mean, cv_std, cv_scores
