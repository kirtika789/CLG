import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping

from utils import set_seeds, Timer
from preprocessing import create_sequences, sort_by_label
from evaluation import (
    calculate_comprehensive_metrics,
    plot_and_save_confusion_matrix,
    plot_and_save_training_curves
)

def create_lstm_model(seq_length, num_features, num_classes, lstm_units=[64, 32], dropout_rate=0.2, learning_rate=0.001, optimizer_name='adam'):
    """
    Constructs a Long Short-Term Memory (LSTM) sequence model.
    """
    model = Sequential()
    model.add(Input(shape=(seq_length, num_features)))
    if len(lstm_units) == 1:
        model.add(LSTM(lstm_units[0], return_sequences=False))
        model.add(Dropout(dropout_rate))
    else:
        model.add(LSTM(lstm_units[0], return_sequences=True))
        model.add(Dropout(dropout_rate))
        for units in lstm_units[1:-1]:
            model.add(LSTM(units, return_sequences=True))
            model.add(Dropout(dropout_rate))
        model.add(LSTM(lstm_units[-1], return_sequences=False))
        model.add(Dropout(dropout_rate))

    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
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

def run_lstm_experiment(dataset_dict, output_dir, seq_length=10):
    """
    Runs hyperparameter tuning, model training, evaluation, and saving for LSTM.
    Sequence Construction: CICIoT2023 tabular flow records are windowed (seq_length=10) 
    independently within each data split to maintain temporal sequence relationships 
    without train/val/test data leakage.
    """
    set_seeds(42)
    print("\n==================== MODEL 6: LONG SHORT-TERM MEMORY (LSTM) ====================")
    print(f"[LSTM SEQUENCE CONSTRUCTION] Constructing sequence windows (seq_length={seq_length}) within train, val, and test splits...")

    X_train_raw, y_train_raw = dataset_dict['X_train'], dataset_dict['y_train']
    X_val_raw, y_val_raw = dataset_dict['X_val'], dataset_dict['y_val']
    X_test_raw, y_test_raw = dataset_dict['X_test'], dataset_dict['y_test']
    class_names = dataset_dict['class_names']
    num_features = dataset_dict['num_features']
    num_classes = dataset_dict['num_classes']
    class_weights = dataset_dict['class_weight_dict']

    # Group each split by class before windowing. This dataset has no
    # genuine per-row timestamp, so a plain sliding window over
    # label-stratified-sampled data would otherwise blend unrelated
    # classes into a single sequence.
    X_train_sorted, y_train_sorted = sort_by_label(X_train_raw, y_train_raw)
    X_val_sorted, y_val_sorted = sort_by_label(X_val_raw, y_val_raw)
    X_test_sorted, y_test_sorted = sort_by_label(X_test_raw, y_test_raw)

    X_train_seq, y_train_seq = create_sequences(X_train_sorted, y_train_sorted, seq_length=seq_length)
    X_val_seq, y_val_seq = create_sequences(X_val_sorted, y_val_sorted, seq_length=seq_length)
    X_test_seq, y_test_seq = create_sequences(X_test_sorted, y_test_sorted, seq_length=seq_length)

    # Search Space for LSTM
    hyperparam_configs = [
        {'units': [64, 32], 'lr': 0.001, 'batch_size': 128, 'opt': 'adam', 'dropout': 0.2},
        {'units': [128, 64], 'lr': 0.001, 'batch_size': 128, 'opt': 'adam', 'dropout': 0.3},
        {'units': [64], 'lr': 0.0005, 'batch_size': 64, 'opt': 'rmsprop', 'dropout': 0.2},
        {'units': [128, 64, 32], 'lr': 0.0001, 'batch_size': 128, 'opt': 'adam', 'dropout': 0.2},
    ]

    tuning_results = []
    best_val_f1 = -1.0
    best_config = None

    print(f"[LSTM TUNING] Testing {len(hyperparam_configs)} hyperparameter configurations...")

    for i, cfg in enumerate(hyperparam_configs):
        print(f"  Config {i+1}/{len(hyperparam_configs)}: Units={cfg['units']}, Opt={cfg['opt']}, LR={cfg['lr']}, Batch={cfg['batch_size']}")
        model = create_lstm_model(
            seq_length=seq_length,
            num_features=num_features,
            num_classes=num_classes,
            lstm_units=cfg['units'],
            dropout_rate=cfg['dropout'],
            learning_rate=cfg['lr'],
            optimizer_name=cfg['opt']
        )

        early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

        hist = model.fit(
            X_train_seq, y_train_seq,
            validation_data=(X_val_seq, y_val_seq),
            epochs=8,
            batch_size=cfg['batch_size'],
            class_weight=class_weights,
            callbacks=[early_stop],
            verbose=0
        )

        val_probs = model.predict(X_val_seq, batch_size=256, verbose=0)
        val_preds = np.argmax(val_probs, axis=1)

        val_metrics, _ = calculate_comprehensive_metrics(y_val_seq, val_preds, val_probs, class_names)

        tuning_results.append({
            'Model': 'LSTM',
            'Optimizer': cfg['opt'],
            'Learning Rate': cfg['lr'],
            'Batch Size': cfg['batch_size'],
            'Layers': len(cfg['units']),
            'Neurons': str(cfg['units']),
            'Dropout': cfg['dropout'],
            'Epochs': len(hist.history['loss']),
            'PCA Components': 'N/A',
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
    tuning_csv_path = os.path.join(output_dir, 'hyperparameter_tuning', 'lstm_tuning_results.csv')
    df_tuning.to_csv(tuning_csv_path, index=False)
    print(f"[LSTM TUNING] Saved all tuning results to {tuning_csv_path}")

    # Retrain best model
    print("\n[LSTM FINAL TRAINING] Retraining best LSTM configuration...")
    final_model = create_lstm_model(
        seq_length=seq_length,
        num_features=num_features,
        num_classes=num_classes,
        lstm_units=best_config['units'],
        dropout_rate=best_config['dropout'],
        learning_rate=best_config['lr'],
        optimizer_name=best_config['opt']
    )
    early_stop = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)

    with Timer() as t_train:
        final_hist = final_model.fit(
            X_train_seq, y_train_seq,
            validation_data=(X_val_seq, y_val_seq),
            epochs=12,
            batch_size=best_config['batch_size'],
            class_weight=class_weights,
            callbacks=[early_stop],
            verbose=1
        )
    training_time = t_train.duration

    # Evaluation on Test set
    print("[LSTM FINAL EVALUATION] Evaluating on untouched test set...")
    with Timer() as t_infer:
        test_probs = final_model.predict(X_test_seq, batch_size=256, verbose=0)
    inference_time = t_infer.duration
    test_preds = np.argmax(test_probs, axis=1)

    test_metrics, cm = calculate_comprehensive_metrics(y_test_seq, test_preds, test_probs, class_names)
    test_metrics['Training_Time'] = float(training_time)
    test_metrics['Inference_Time'] = float(inference_time)
    test_metrics['Model'] = 'LSTM'

    print(f"[LSTM RESULTS] Test Accuracy: {test_metrics['Accuracy']:.4f} | Precision: {test_metrics['Precision']:.4f} | Recall: {test_metrics['Recall']:.4f} | F1: {test_metrics['F1_Score']:.4f}")

    # Save metrics
    df_metrics = pd.DataFrame([test_metrics])
    df_metrics.to_csv(os.path.join(output_dir, 'metrics', 'lstm_metrics.csv'), index=False)

    # Save confusion matrix
    cm_png_path = os.path.join(output_dir, 'confusion_matrices', 'lstm_confusion_matrix.png')
    cm_csv_path = os.path.join(output_dir, 'confusion_matrices', 'lstm_confusion_matrix.csv')
    plot_and_save_confusion_matrix(cm, class_names, cm_png_path, cm_csv_path, model_name="LSTM")

    # Save training curves
    acc_curve_path = os.path.join(output_dir, 'training_curves', 'lstm_accuracy.png')
    loss_curve_path = os.path.join(output_dir, 'training_curves', 'lstm_loss.png')
    plot_and_save_training_curves(final_hist, acc_curve_path, loss_curve_path, model_name="LSTM")

    # Save predictions
    y_test_labels = [class_names[i] for i in y_test_seq]
    pred_labels = [class_names[i] for i in test_preds]
    max_probs = np.max(test_probs, axis=1)

    df_preds = pd.DataFrame({
        'Actual_Label': y_test_labels,
        'Predicted_Label': pred_labels,
        'Prediction_Probability': max_probs
    })
    df_preds.to_csv(os.path.join(output_dir, 'predictions', 'lstm_predictions.csv'), index=False)

    return test_metrics, best_config