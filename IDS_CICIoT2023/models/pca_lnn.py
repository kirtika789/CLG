import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping

from utils import set_seeds, Timer
from preprocessing import apply_pca_preprocessing
from evaluation import (
    calculate_comprehensive_metrics,
    plot_and_save_confusion_matrix,
    plot_and_save_training_curves
)

def create_lnn_model(input_dim, num_classes, units=128, dropout_rate=0.1, learning_rate=0.001, optimizer_name='adam'):
    """
    Constructs a Lightweight Neural Network (LNN) model.
    LNN Architecture Definition:
    A minimal 1-hidden-layer dense representation architecture designed for low-latency,
    high-throughput network packet classification following PCA dimensionality reduction.
    """
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(units, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(num_classes, activation='softmax')
    ])

    if optimizer_name.lower() == 'rmsprop':
        opt = RMSprop(learning_rate=learning_rate)
    else:
        opt = Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=opt,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def run_pca_lnn_experiment(dataset_dict, output_dir):
    """
    Runs PCA transformation, hyperparameter tuning, model training,
    evaluation, and saving for PCA + LNN.
    PCA fitted ONLY on X_train.
    """
    set_seeds(42)
    print("\n==================== MODEL 4: PCA + LNN (LIGHTWEIGHT NEURAL NETWORK) ====================")

    X_train_raw = dataset_dict['X_train']
    X_val_raw = dataset_dict['X_val']
    X_test_raw = dataset_dict['X_test']
    y_train, y_val, y_test = dataset_dict['y_train'], dataset_dict['y_val'], dataset_dict['y_test']
    class_names = dataset_dict['class_names']
    num_classes = dataset_dict['num_classes']
    class_weights = dataset_dict['class_weight_dict']

    # PCA Analysis (Variance target: 95%)
    print("[PCA ANALYSIS] Fitting PCA ONLY on training split...")
    X_train_pca, X_val_pca, X_test_pca, pca_obj, pca_stats = apply_pca_preprocessing(
        X_train_raw, X_val_raw, X_test_raw, n_components=0.95
    )

    n_pca_components = pca_stats['n_components']
    print(f"[PCA ANALYSIS] Original Features: {pca_stats['original_features']} -> Selected PCA Components: {n_pca_components}")
    print(f"[PCA ANALYSIS] Total Explained Variance Ratio: {pca_stats['total_explained_variance']*100:.2f}%")

    # Save PCA Explained Variance CSV & Plot
    df_pca_var = pd.DataFrame({
        'Component': [f"PC_{i+1}" for i in range(n_pca_components)],
        'Explained_Variance_Ratio': pca_stats['explained_variance_ratio'],
        'Cumulative_Explained_Variance': pca_stats['cumulative_explained_variance']
    })
    pca_csv_path = os.path.join(output_dir, 'feature_analysis', 'pca_explained_variance.csv')
    df_pca_var.to_csv(pca_csv_path, index=False)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.bar(range(1, n_pca_components + 1), pca_stats['explained_variance_ratio'], alpha=0.6, color='b', label='Individual Variance')
    ax1.set_xlabel('Principal Component Index', fontsize=12)
    ax1.set_ylabel('Individual Explained Variance', color='b', fontsize=12)
    
    ax2 = ax1.twinx()
    ax2.plot(range(1, n_pca_components + 1), pca_stats['cumulative_explained_variance'], 'r-o', label='Cumulative Variance', linewidth=2)
    ax2.set_ylabel('Cumulative Explained Variance', color='r', fontsize=12)
    ax1.set_title(f'PCA Explained Variance ({n_pca_components} Components = {pca_stats["total_explained_variance"]*100:.1f}%)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    pca_png_path = os.path.join(output_dir, 'feature_analysis', 'pca_explained_variance.png')
    plt.savefig(pca_png_path, dpi=300)
    plt.close(fig)
    print(f"[FEATURE ANALYSIS] Saved PCA variance analysis to {pca_csv_path} and {pca_png_path}")

    # Search Space for LNN
    hyperparam_configs = [
        {'units': 128, 'lr': 0.001, 'batch_size': 128, 'opt': 'adam', 'dropout': 0.1},
        {'units': 256, 'lr': 0.001, 'batch_size': 128, 'opt': 'adam', 'dropout': 0.2},
        {'units': 64, 'lr': 0.0005, 'batch_size': 64, 'opt': 'rmsprop', 'dropout': 0.1},
        {'units': 128, 'lr': 0.0001, 'batch_size': 256, 'opt': 'adam', 'dropout': 0.1},
    ]

    tuning_results = []
    best_val_f1 = -1.0
    best_config = None

    print(f"[PCA+LNN TUNING] Testing {len(hyperparam_configs)} hyperparameter configurations...")

    for i, cfg in enumerate(hyperparam_configs):
        print(f"  Config {i+1}/{len(hyperparam_configs)}: Units={cfg['units']}, Opt={cfg['opt']}, LR={cfg['lr']}, Batch={cfg['batch_size']}")
        model = create_lnn_model(
            input_dim=n_pca_components,
            num_classes=num_classes,
            units=cfg['units'],
            dropout_rate=cfg['dropout'],
            learning_rate=cfg['lr'],
            optimizer_name=cfg['opt']
        )

        early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

        hist = model.fit(
            X_train_pca, y_train,
            validation_data=(X_val_pca, y_val),
            epochs=10,
            batch_size=cfg['batch_size'],
            class_weight=class_weights,
            callbacks=[early_stop],
            verbose=0
        )

        val_probs = model.predict(X_val_pca, batch_size=256, verbose=0)
        val_preds = np.argmax(val_probs, axis=1)

        val_metrics, _ = calculate_comprehensive_metrics(y_val, val_preds, val_probs, class_names)

        tuning_results.append({
            'Model': 'PCA + LNN',
            'Optimizer': cfg['opt'],
            'Learning Rate': cfg['lr'],
            'Batch Size': cfg['batch_size'],
            'Layers': 1,
            'Neurons': str(cfg['units']),
            'Dropout': cfg['dropout'],
            'Epochs': len(hist.history['loss']),
            'PCA Components': n_pca_components,
            'Validation Accuracy': val_metrics['Accuracy'],
            'Validation Precision': val_metrics['Precision'],
            'Validation Recall': val_metrics['Recall'],
            'Validation F1': val_metrics['F1_Score']
        })

        if val_metrics['F1_Score'] > best_val_f1:
            best_val_f1 = val_metrics['F1_Score']
            best_config = cfg

    # Save tuning CSV
    df_tuning = pd.DataFrame(tuning_results)
    tuning_csv_path = os.path.join(output_dir, 'hyperparameter_tuning', 'pca_lnn_tuning_results.csv')
    df_tuning.to_csv(tuning_csv_path, index=False)
    print(f"[PCA+LNN TUNING] Saved all tuning results to {tuning_csv_path}")

    # Retrain best model
    print("\n[PCA+LNN FINAL TRAINING] Retraining best LNN configuration...")
    final_model = create_lnn_model(
        input_dim=n_pca_components,
        num_classes=num_classes,
        units=best_config['units'],
        dropout_rate=best_config['dropout'],
        learning_rate=best_config['lr'],
        optimizer_name=best_config['opt']
    )
    early_stop = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)

    with Timer() as t_train:
        final_hist = final_model.fit(
            X_train_pca, y_train,
            validation_data=(X_val_pca, y_val),
            epochs=15,
            batch_size=best_config['batch_size'],
            class_weight=class_weights,
            callbacks=[early_stop],
            verbose=1
        )
    training_time = t_train.duration

    # Evaluation on Test set
    print("[PCA+LNN FINAL EVALUATION] Evaluating on untouched test set...")
    with Timer() as t_infer:
        test_probs = final_model.predict(X_test_pca, batch_size=256, verbose=0)
    inference_time = t_infer.duration
    test_preds = np.argmax(test_probs, axis=1)

    test_metrics, cm = calculate_comprehensive_metrics(y_test, test_preds, test_probs, class_names)
    test_metrics['Training_Time'] = float(training_time)
    test_metrics['Inference_Time'] = float(inference_time)
    test_metrics['Model'] = 'PCA + LNN'

    print(f"[PCA+LNN RESULTS] Test Accuracy: {test_metrics['Accuracy']:.4f} | Precision: {test_metrics['Precision']:.4f} | Recall: {test_metrics['Recall']:.4f} | F1: {test_metrics['F1_Score']:.4f}")

    # Save metrics
    df_metrics = pd.DataFrame([test_metrics])
    df_metrics.to_csv(os.path.join(output_dir, 'metrics', 'pca_lnn_metrics.csv'), index=False)

    # Save confusion matrix
    cm_png_path = os.path.join(output_dir, 'confusion_matrices', 'pca_lnn_confusion_matrix.png')
    cm_csv_path = os.path.join(output_dir, 'confusion_matrices', 'pca_lnn_confusion_matrix.csv')
    plot_and_save_confusion_matrix(cm, class_names, cm_png_path, cm_csv_path, model_name="PCA + LNN")

    # Save training curves
    acc_curve_path = os.path.join(output_dir, 'training_curves', 'pca_lnn_accuracy.png')
    loss_curve_path = os.path.join(output_dir, 'training_curves', 'pca_lnn_loss.png')
    plot_and_save_training_curves(final_hist, acc_curve_path, loss_curve_path, model_name="PCA + LNN")

    # Save predictions
    y_test_labels = [class_names[i] for i in y_test]
    pred_labels = [class_names[i] for i in test_preds]
    max_probs = np.max(test_probs, axis=1)

    df_preds = pd.DataFrame({
        'Actual_Label': y_test_labels,
        'Predicted_Label': pred_labels,
        'Prediction_Probability': max_probs
    })
    df_preds.to_csv(os.path.join(output_dir, 'predictions', 'pca_lnn_predictions.csv'), index=False)

    return test_metrics, best_config
