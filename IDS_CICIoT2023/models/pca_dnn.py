import os
import time
import numpy as np
import pandas as pd
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

def create_pca_dnn_model(input_dim, num_classes, layers_config=[256, 128], dropout_rate=0.2, learning_rate=0.001, optimizer_name='adam'):
    """
    Constructs a Deep Neural Network model for PCA-transformed feature inputs.
    """
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(layers_config[0], activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    for units in layers_config[1:]:
        model.add(Dense(units, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))

    model.add(Dense(num_classes, activation='softmax'))

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

def run_pca_dnn_experiment(dataset_dict, output_dir):
    """
    Runs PCA transformation, hyperparameter tuning, model training,
    evaluation, and saving for PCA + DNN.
    PCA fitted ONLY on X_train.
    """
    set_seeds(42)
    print("\n==================== MODEL 5: PCA + DEEP NEURAL NETWORK (PCA + DNN) ====================")

    X_train_raw = dataset_dict['X_train']
    X_val_raw = dataset_dict['X_val']
    X_test_raw = dataset_dict['X_test']
    y_train, y_val, y_test = dataset_dict['y_train'], dataset_dict['y_val'], dataset_dict['y_test']
    class_names = dataset_dict['class_names']
    num_classes = dataset_dict['num_classes']
    class_weights = dataset_dict['class_weight_dict']

    # Apply PCA (Variance target: 95%)
    print("[PCA ANALYSIS] Transforming features via PCA fitted on X_train...")
    X_train_pca, X_val_pca, X_test_pca, pca_obj, pca_stats = apply_pca_preprocessing(
        X_train_raw, X_val_raw, X_test_raw, n_components=0.95
    )
    n_pca_components = pca_stats['n_components']

    # Search Space for PCA + DNN
    hyperparam_configs = [
        {'layers': [256, 128], 'lr': 0.001, 'batch_size': 128, 'opt': 'adam', 'dropout': 0.2},
        {'layers': [512, 256, 128], 'lr': 0.001, 'batch_size': 128, 'opt': 'adam', 'dropout': 0.3},
        {'layers': [128, 64], 'lr': 0.0005, 'batch_size': 64, 'opt': 'rmsprop', 'dropout': 0.2},
        {'layers': [256, 128, 64], 'lr': 0.0001, 'batch_size': 128, 'opt': 'adam', 'dropout': 0.2},
    ]

    tuning_results = []
    best_val_f1 = -1.0
    best_config = None

    print(f"[PCA+DNN TUNING] Testing {len(hyperparam_configs)} hyperparameter configurations...")

    for i, cfg in enumerate(hyperparam_configs):
        print(f"  Config {i+1}/{len(hyperparam_configs)}: Layers={cfg['layers']}, Opt={cfg['opt']}, LR={cfg['lr']}, Batch={cfg['batch_size']}")
        model = create_pca_dnn_model(
            input_dim=n_pca_components,
            num_classes=num_classes,
            layers_config=cfg['layers'],
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
            'Model': 'PCA + DNN',
            'Optimizer': cfg['opt'],
            'Learning Rate': cfg['lr'],
            'Batch Size': cfg['batch_size'],
            'Layers': len(cfg['layers']),
            'Neurons': str(cfg['layers']),
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
    tuning_csv_path = os.path.join(output_dir, 'hyperparameter_tuning', 'pca_dnn_tuning_results.csv')
    df_tuning.to_csv(tuning_csv_path, index=False)
    print(f"[PCA+DNN TUNING] Saved all tuning results to {tuning_csv_path}")

    # Retrain best model
    print("\n[PCA+DNN FINAL TRAINING] Retraining best PCA+DNN configuration...")
    final_model = create_pca_dnn_model(
        input_dim=n_pca_components,
        num_classes=num_classes,
        layers_config=best_config['layers'],
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
    print("[PCA+DNN FINAL EVALUATION] Evaluating on untouched test set...")
    with Timer() as t_infer:
        test_probs = final_model.predict(X_test_pca, batch_size=256, verbose=0)
    inference_time = t_infer.duration
    test_preds = np.argmax(test_probs, axis=1)

    test_metrics, cm = calculate_comprehensive_metrics(y_test, test_preds, test_probs, class_names)
    test_metrics['Training_Time'] = float(training_time)
    test_metrics['Inference_Time'] = float(inference_time)
    test_metrics['Model'] = 'PCA + DNN'

    print(f"[PCA+DNN RESULTS] Test Accuracy: {test_metrics['Accuracy']:.4f} | Precision: {test_metrics['Precision']:.4f} | Recall: {test_metrics['Recall']:.4f} | F1: {test_metrics['F1_Score']:.4f}")

    # Save metrics
    df_metrics = pd.DataFrame([test_metrics])
    df_metrics.to_csv(os.path.join(output_dir, 'metrics', 'pca_dnn_metrics.csv'), index=False)

    # Save confusion matrix
    cm_png_path = os.path.join(output_dir, 'confusion_matrices', 'pca_dnn_confusion_matrix.png')
    cm_csv_path = os.path.join(output_dir, 'confusion_matrices', 'pca_dnn_confusion_matrix.csv')
    plot_and_save_confusion_matrix(cm, class_names, cm_png_path, cm_csv_path, model_name="PCA + DNN")

    # Save training curves
    acc_curve_path = os.path.join(output_dir, 'training_curves', 'pca_dnn_accuracy.png')
    loss_curve_path = os.path.join(output_dir, 'training_curves', 'pca_dnn_loss.png')
    plot_and_save_training_curves(final_hist, acc_curve_path, loss_curve_path, model_name="PCA + DNN")

    # Save predictions
    y_test_labels = [class_names[i] for i in y_test]
    pred_labels = [class_names[i] for i in test_preds]
    max_probs = np.max(test_probs, axis=1)

    df_preds = pd.DataFrame({
        'Actual_Label': y_test_labels,
        'Predicted_Label': pred_labels,
        'Prediction_Probability': max_probs
    })
    df_preds.to_csv(os.path.join(output_dir, 'predictions', 'pca_dnn_predictions.csv'), index=False)

    return test_metrics, best_config
