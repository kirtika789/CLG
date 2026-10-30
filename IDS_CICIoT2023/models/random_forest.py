import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from utils import set_seeds, Timer
from tuning.hyperparams import RF_GRID
from evaluation import (
    calculate_comprehensive_metrics,
    plot_and_save_confusion_matrix,
    plot_and_save_roc_curves,
    save_complete_predictions,
    perform_5_fold_cv
)

def align_probabilities(probs, classes_seen, num_classes):
    """
    Ensures predicted probability matrix aligns with the full class spectrum (num_classes).
    Maps columns according to classes seen during training, and fills unobserved classes with 0.0.
    """
    if probs is None:
        return None
    if probs.shape[1] == num_classes:
        return probs
    aligned_probs = np.zeros((len(probs), num_classes), dtype=np.float64)
    for col_idx, class_idx in enumerate(classes_seen):
        if class_idx < num_classes:
            aligned_probs[:, class_idx] = probs[:, col_idx]
    return aligned_probs

def run_random_forest_experiment(dataset_dict, output_dir, hyperparam_configs=None):
    """
    Runs hyperparameter tuning, model training, feature importance analysis,
    evaluation, and output export for Random Forest.
    """
    set_seeds(42)
    print("\n==================== MODEL 3: RANDOM FOREST ====================")

    # Ensure all required output subdirectories exist
    tuning_dir = os.path.join(output_dir, 'hyperparameter_tuning')
    metrics_dir = os.path.join(output_dir, 'metrics')
    cm_dir = os.path.join(output_dir, 'confusion_matrices')
    feat_dir = os.path.join(output_dir, 'feature_analysis')
    pred_dir = os.path.join(output_dir, 'predictions')
    comp_dir = os.path.join(output_dir, 'model_comparison')

    for d in [tuning_dir, metrics_dir, cm_dir, feat_dir, pred_dir, comp_dir]:
        os.makedirs(d, exist_ok=True)

    X_train, y_train = dataset_dict['X_train'], dataset_dict['y_train']
    X_val, y_val = dataset_dict['X_val'], dataset_dict['y_val']
    X_test, y_test = dataset_dict['X_test'], dataset_dict['y_test']
    class_names = dataset_dict['class_names']
    feature_names = dataset_dict.get('feature_names', [f"Feature_{i}" for i in range(X_train.shape[1])])
    num_classes = len(class_names)

    # Search space from hyperparams or passed configs
    if hyperparam_configs is None:
        hyperparam_configs = RF_GRID

    tuning_results = []
    best_val_f1 = -1.0
    best_config = None
    best_rf_model = None

    print(f"[RANDOM FOREST TUNING] Testing {len(hyperparam_configs)} hyperparameter configurations...")

    for i, cfg in enumerate(hyperparam_configs):
        print(f"  Config {i+1}/{len(hyperparam_configs)}: Trees={cfg['n_estimators']}, MaxDepth={cfg['max_depth']}, Split={cfg['min_samples_split']}, MaxFeats={cfg['max_features']}")
        
        rf = RandomForestClassifier(
            n_estimators=cfg['n_estimators'],
            max_depth=cfg['max_depth'],
            min_samples_split=cfg['min_samples_split'],
            max_features=cfg['max_features'],
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        rf.fit(X_train, y_train)

        val_probs = rf.predict_proba(X_val)
        val_probs = align_probabilities(val_probs, rf.classes_, num_classes)
        val_preds = rf.predict(X_val)

        val_metrics, _ = calculate_comprehensive_metrics(y_val, val_preds, val_probs, class_names)

        tuning_results.append({
            'Model': 'Random Forest',
            'Optimizer': 'N/A (Tree Ensemble)',
            'Learning Rate': 'N/A',
            'Batch Size': 'N/A',
            'Layers': cfg['n_estimators'],
            'Neurons': f"max_depth={cfg['max_depth']}",
            'Dropout': 'N/A',
            'Epochs': cfg['n_estimators'],
            'PCA Components': 'N/A',
            'Validation Accuracy': val_metrics['Accuracy'],
            'Validation Precision': val_metrics['Precision'],
            'Validation Recall': val_metrics['Recall'],
            'Validation F1': val_metrics['F1_Score']
        })

        if val_metrics['F1_Score'] > best_val_f1:
            best_val_f1 = val_metrics['F1_Score']
            best_config = cfg
            best_rf_model = rf

    # Fallback if no config improved score
    if best_config is None and len(hyperparam_configs) > 0:
        best_config = hyperparam_configs[0]

    # Save tuning CSV
    df_tuning = pd.DataFrame(tuning_results)
    tuning_csv_path = os.path.join(tuning_dir, 'random_forest_tuning_results.csv')
    df_tuning.to_csv(tuning_csv_path, index=False)
    print(f"[RANDOM FOREST TUNING] Saved all tuning results to {tuning_csv_path}")

    # Retrain best model cleanly to measure exact timing
    print("\n[RANDOM FOREST FINAL TRAINING] Training best Random Forest model...")
    final_rf = RandomForestClassifier(
        n_estimators=best_config['n_estimators'],
        max_depth=best_config['max_depth'],
        min_samples_split=best_config['min_samples_split'],
        max_features=best_config['max_features'],
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    with Timer() as t_train:
        final_rf.fit(X_train, y_train)
    training_time = t_train.duration

    # Evaluation on Test set
    print("[RANDOM FOREST FINAL EVALUATION] Evaluating on untouched test set...")
    with Timer() as t_infer:
        test_probs = final_rf.predict_proba(X_test)
        test_probs = align_probabilities(test_probs, final_rf.classes_, num_classes)
        test_preds = final_rf.predict(X_test)
    inference_time = t_infer.duration

    test_metrics, cm = calculate_comprehensive_metrics(y_test, test_preds, test_probs, class_names)
    test_metrics['Training_Time'] = float(training_time)
    test_metrics['Inference_Time'] = float(inference_time)
    test_metrics['Model'] = 'Random Forest'

    # Compute Training, Validation, and 5-Fold Cross-Validation Accuracies
    print("[RANDOM FOREST ACCURACY EVALUATION] Calculating training and validation accuracies...")
    train_preds = final_rf.predict(X_train)
    train_accuracy = float(accuracy_score(y_train, train_preds))

    val_preds = final_rf.predict(X_val)
    val_accuracy = float(accuracy_score(y_val, val_preds))

    print("[RANDOM FOREST 5-FOLD CV] Running 5-fold stratified cross-validation...")
    cv_mean, cv_std, _ = perform_5_fold_cv(
        model_or_builder=final_rf,
        X=X_train,
        y=y_train,
        is_keras=False,
        model_name="Random Forest"
    )

    test_metrics['Train_Accuracy'] = train_accuracy
    test_metrics['Val_Accuracy'] = val_accuracy
    test_metrics['CV_Accuracy'] = cv_mean
    test_metrics['CV_Std'] = cv_std
    test_metrics['Test_Accuracy'] = test_metrics['Accuracy']

    print(f"[RANDOM FOREST RESULTS] Test Accuracy: {test_metrics['Accuracy']:.4f} | Train Accuracy: {train_accuracy:.4f} | Val Accuracy: {val_accuracy:.4f} | 5-Fold CV Acc: {cv_mean:.4f} (±{cv_std:.4f})")
    print(f"[RANDOM FOREST METRICS] Precision: {test_metrics['Precision']:.4f} | Recall: {test_metrics['Recall']:.4f} | F1: {test_metrics['F1_Score']:.4f}")

    # Save metrics
    df_metrics = pd.DataFrame([test_metrics])
    df_metrics.to_csv(os.path.join(metrics_dir, 'random_forest_metrics.csv'), index=False)

    # Save confusion matrix
    cm_png_path = os.path.join(cm_dir, 'random_forest_confusion_matrix.png')
    cm_csv_path = os.path.join(cm_dir, 'random_forest_confusion_matrix.csv')
    plot_and_save_confusion_matrix(cm, class_names, cm_png_path, cm_csv_path, model_name="Random Forest")

    # Feature Importance Analysis
    importances = final_rf.feature_importances_
    feat_names = feature_names if len(feature_names) == len(importances) else [f'Feature_{i}' for i in range(len(importances))]
    df_feat = pd.DataFrame({
        'Feature': feat_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)

    feat_csv_path = os.path.join(feat_dir, 'random_forest_feature_importance.csv')
    df_feat.to_csv(feat_csv_path, index=False)

    # Plot Top 20 Features
    fig, ax = plt.subplots(figsize=(10, 8))
    top20 = df_feat.head(20)
    sns.barplot(x='Importance', y='Feature', data=top20, hue='Feature', palette='viridis', legend=False, ax=ax)
    ax.set_title('Random Forest — Top 20 Feature Importances', fontsize=14, fontweight='bold')
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_ylabel('Feature Name', fontsize=12)
    plt.tight_layout()
    top_feat_png_path = os.path.join(feat_dir, 'random_forest_top_features.png')
    plt.savefig(top_feat_png_path, dpi=300)
    plt.close(fig)

    print(f"[FEATURE ANALYSIS] Saved Random Forest feature importances to {feat_csv_path} and {top_feat_png_path}")

    # Save Predictions with Complete Probability Matrix
    preds_csv_path = os.path.join(pred_dir, 'random_forest_predictions.csv')
    save_complete_predictions(y_test, test_preds, test_probs, class_names, preds_csv_path, model_name="Random Forest")

    # Plot & Save One-vs-Rest ROC Curves
    roc_png_path = os.path.join(comp_dir, 'random_forest_roc_curve.png')
    plot_and_save_roc_curves(y_test, test_probs, class_names, roc_png_path, model_name="Random Forest")

    return test_metrics, best_config
