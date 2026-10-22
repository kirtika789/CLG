import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

from utils import set_seeds, Timer
from evaluation import (
    calculate_comprehensive_metrics,
    plot_and_save_confusion_matrix
)

def run_random_forest_experiment(dataset_dict, output_dir):
    """
    Runs hyperparameter tuning, model training, feature importance analysis,
    evaluation, and output export for Random Forest.
    """
    set_seeds(42)
    print("\n==================== MODEL 3: RANDOM FOREST ====================")

    X_train, y_train = dataset_dict['X_train'], dataset_dict['y_train']
    X_val, y_val = dataset_dict['X_val'], dataset_dict['y_val']
    X_test, y_test = dataset_dict['X_test'], dataset_dict['y_test']
    class_names = dataset_dict['class_names']
    feature_names = dataset_dict['feature_names']

    # Search space
    hyperparam_configs = [
        {'n_estimators': 100, 'max_depth': 20, 'min_samples_split': 2, 'max_features': 'sqrt'},
        {'n_estimators': 200, 'max_depth': 30, 'min_samples_split': 5, 'max_features': 'sqrt'},
        {'n_estimators': 150, 'max_depth': None, 'min_samples_split': 2, 'max_features': 'log2'},
        {'n_estimators': 250, 'max_depth': 25, 'min_samples_split': 4, 'max_features': 'sqrt'},
    ]

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

    # Save tuning CSV
    df_tuning = pd.DataFrame(tuning_results)
    tuning_csv_path = os.path.join(output_dir, 'hyperparameter_tuning', 'random_forest_tuning_results.csv')
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
        test_preds = final_rf.predict(X_test)
    inference_time = t_infer.duration

    test_metrics, cm = calculate_comprehensive_metrics(y_test, test_preds, test_probs, class_names)
    test_metrics['Training_Time'] = float(training_time)
    test_metrics['Inference_Time'] = float(inference_time)
    test_metrics['Model'] = 'Random Forest'

    print(f"[RANDOM FOREST RESULTS] Test Accuracy: {test_metrics['Accuracy']:.4f} | Precision: {test_metrics['Precision']:.4f} | Recall: {test_metrics['Recall']:.4f} | F1: {test_metrics['F1_Score']:.4f}")

    # Save metrics
    df_metrics = pd.DataFrame([test_metrics])
    df_metrics.to_csv(os.path.join(output_dir, 'metrics', 'random_forest_metrics.csv'), index=False)

    # Save confusion matrix
    cm_png_path = os.path.join(output_dir, 'confusion_matrices', 'random_forest_confusion_matrix.png')
    cm_csv_path = os.path.join(output_dir, 'confusion_matrices', 'random_forest_confusion_matrix.csv')
    plot_and_save_confusion_matrix(cm, class_names, cm_png_path, cm_csv_path, model_name="Random Forest")

    # Feature Importance Analysis
    importances = final_rf.feature_importances_
    df_feat = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)

    feat_csv_path = os.path.join(output_dir, 'feature_analysis', 'random_forest_feature_importance.csv')
    df_feat.to_csv(feat_csv_path, index=False)

    # Plot Top 20 Features
    fig, ax = plt.subplots(figsize=(10, 8))
    top20 = df_feat.head(20)
    sns.barplot(x='Importance', y='Feature', data=top20, palette='viridis', ax=ax)
    ax.set_title('Random Forest — Top 20 Feature Importances', fontsize=14, fontweight='bold')
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_ylabel('Feature Name', fontsize=12)
    plt.tight_layout()
    top_feat_png_path = os.path.join(output_dir, 'feature_analysis', 'random_forest_top_features.png')
    plt.savefig(top_feat_png_path, dpi=300)
    plt.close(fig)

    print(f"[FEATURE ANALYSIS] Saved Random Forest feature importances to {feat_csv_path} and {top_feat_png_path}")

    # Save predictions
    y_test_labels = [class_names[i] for i in y_test]
    pred_labels = [class_names[i] for i in test_preds]
    max_probs = np.max(test_probs, axis=1)

    df_preds = pd.DataFrame({
        'Actual_Label': y_test_labels,
        'Predicted_Label': pred_labels,
        'Prediction_Probability': max_probs
    })
    df_preds.to_csv(os.path.join(output_dir, 'predictions', 'random_forest_predictions.csv'), index=False)

    return test_metrics, best_config
