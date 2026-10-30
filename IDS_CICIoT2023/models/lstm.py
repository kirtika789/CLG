import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight

from utils import set_seeds, Timer
from sklearn.metrics import accuracy_score
from preprocessing import create_sequences
from evaluation import (
    calculate_comprehensive_metrics,
    plot_and_save_confusion_matrix,
    plot_and_save_training_curves,
    plot_and_save_roc_curves,
    save_complete_predictions,
    perform_5_fold_cv
)

def create_lstm_model(
    seq_length,
    num_features,
    num_classes,
    lstm_units=[128, 64],
    dense_units=128,
    dropout_rate=0.3,
    learning_rate=0.0005,
    optimizer_name='adam'
):
    """
    Constructs an improved Long Short-Term Memory (LSTM) network architecture.
    
    Architecture Design:
    - Input tensor: (batch_size, seq_length, num_features)
    - 1-layer or 2-layer stacked LSTM with recurrent regularization.
    - Dropout for inter-layer and dense regularization.
    - Dense projection head with ReLU activation.
    - Batch Normalization to stabilize latent representation activations.
    - 34-unit Softmax output layer with sparse categorical cross-entropy.
    """
    model = Sequential()
    model.add(Input(shape=(seq_length, num_features)))

    if len(lstm_units) == 1:
        model.add(LSTM(lstm_units[0], return_sequences=False, dropout=0.1))
        model.add(Dropout(dropout_rate))
    else:
        for units in lstm_units[:-1]:
            model.add(LSTM(units, return_sequences=True, dropout=0.1))
            model.add(Dropout(dropout_rate))
        model.add(LSTM(lstm_units[-1], return_sequences=False, dropout=0.1))
        model.add(Dropout(dropout_rate))

    # Dense representation head
    model.add(Dense(dense_units, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate * 0.75))
    model.add(Dense(num_classes, activation='softmax'))

    # Optimizer
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

def compute_stabilized_class_weights(y_train, max_weight=35.0):
    """
    Computes class weights from training labels with upper-bound clamping.
    Extreme class imbalances (e.g. 10,000x) can destabilize recurrent gates.
    Clamping to a reasonable upper bound protects against gradient explosions
    while still heavily penalizing rare attack misclassifications.
    """
    classes = np.unique(y_train)
    raw_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    clamped_weights = {
        int(c): float(np.clip(w, 0.5, max_weight))
        for c, w in zip(classes, raw_weights)
    }
    return clamped_weights

def run_lstm_experiment(dataset_dict, output_dir, seq_length=10):
    """
    Runs hyperparameter tuning, model training, evaluation, and saving for improved LSTM.
    
    Methodological Integrity:
    - Sequence Construction: Sequences are constructed in chronological / natural arrival order 
      within each split independently. Records are NEVER artificially grouped or sorted by label.
    - Zero Data Leakage: Scalers, label encoders, sequence windows, and class statistics are 
      derived strictly from training data without crossing split boundaries.
    """
    set_seeds(42)
    print("\n==================== MODEL 6: IMPROVED LONG SHORT-TERM MEMORY (LSTM) ====================")
    print(f"[LSTM SEQUENCE CONSTRUCTION] Constructing natural sequence windows (seq_length={seq_length})...")
    print("  [NOTE] Natural sequence ordering preserved. No artificial label sorting applied.")

    X_train_raw, y_train_raw = dataset_dict['X_train'], dataset_dict['y_train']
    X_val_raw, y_val_raw = dataset_dict['X_val'], dataset_dict['y_val']
    X_test_raw, y_test_raw = dataset_dict['X_test'], dataset_dict['y_test']
    class_names = dataset_dict['class_names']
    num_features = dataset_dict['num_features']
    num_classes = dataset_dict['num_classes']

    # Construct sequences in natural split order without sort_by_label
    X_train_seq, y_train_seq = create_sequences(X_train_raw, y_train_raw, seq_length=seq_length)
    X_val_seq, y_val_seq = create_sequences(X_val_raw, y_val_raw, seq_length=seq_length)
    X_test_seq, y_test_seq = create_sequences(X_test_raw, y_test_raw, seq_length=seq_length)

    print(f"  Sequence Tensors: Train={X_train_seq.shape}, Val={X_val_seq.shape}, Test={X_test_seq.shape}")

    # Compute stabilized class weights exclusively from training sequences
    class_weights = compute_stabilized_class_weights(y_train_seq, max_weight=35.0)

    # Search Space for Improved LSTM (Tuning strictly on validation split)
    hyperparam_configs = [
        {'name': '1-Layer LSTM (128 units)', 'units': [128], 'dense': 128, 'lr': 0.001, 'batch_size': 64, 'opt': 'adam', 'dropout': 0.25},
        {'name': '1-Layer LSTM (128 units, low LR)', 'units': [128], 'dense': 128, 'lr': 0.0005, 'batch_size': 64, 'opt': 'adam', 'dropout': 0.3},
        {'name': '2-Layer Hierarchical LSTM (128->64)', 'units': [128, 64], 'dense': 128, 'lr': 0.0005, 'batch_size': 64, 'opt': 'adam', 'dropout': 0.25},
        {'name': '1-Layer LSTM (64 units, RMSprop)', 'units': [64], 'dense': 64, 'lr': 0.0005, 'batch_size': 64, 'opt': 'rmsprop', 'dropout': 0.2},
    ]

    tuning_results = []
    best_val_f1 = -1.0
    best_config = None

    print(f"\n[LSTM TUNING] Testing {len(hyperparam_configs)} hyperparameter configurations on validation split...")

    for i, cfg in enumerate(hyperparam_configs):
        print(f"\n  Config {i+1}/{len(hyperparam_configs)}: {cfg['name']} | Units={cfg['units']}, Opt={cfg['opt']}, LR={cfg['lr']}, Batch={cfg['batch_size']}")
        model = create_lstm_model(
            seq_length=seq_length,
            num_features=num_features,
            num_classes=num_classes,
            lstm_units=cfg['units'],
            dense_units=cfg['dense'],
            dropout_rate=cfg['dropout'],
            learning_rate=cfg['lr'],
            optimizer_name=cfg['opt']
        )

        early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5, verbose=0)

        hist = model.fit(
            X_train_seq, y_train_seq,
            validation_data=(X_val_seq, y_val_seq),
            epochs=8,
            batch_size=cfg['batch_size'],
            class_weight=class_weights,
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )

        val_probs = model.predict(X_val_seq, batch_size=256, verbose=0)
        val_preds = np.argmax(val_probs, axis=1)

        val_metrics, _ = calculate_comprehensive_metrics(y_val_seq, val_preds, val_probs, class_names)
        print(f"    Val Accuracy: {val_metrics['Accuracy']:.4f} | Val F1: {val_metrics['F1_Score']:.4f} | Epochs: {len(hist.history['loss'])}")

        tuning_results.append({
            'Model': 'LSTM',
            'Configuration': cfg['name'],
            'Optimizer': cfg['opt'],
            'Learning Rate': cfg['lr'],
            'Batch Size': cfg['batch_size'],
            'Layers': len(cfg['units']),
            'Neurons': str(cfg['units']),
            'Dense Units': cfg['dense'],
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
    print(f"\n[LSTM TUNING] Saved all tuning results to {tuning_csv_path}")
    print(f"[LSTM TUNING] Selected Best Config: {best_config['name']} (Validation F1: {best_val_f1:.4f})")

    # Retrain best model with full epochs, dynamic LR reduction, and ModelCheckpoint
    print(f"\n[LSTM FINAL TRAINING] Retraining winning architecture ({best_config['name']})...")
    final_model = create_lstm_model(
        seq_length=seq_length,
        num_features=num_features,
        num_classes=num_classes,
        lstm_units=best_config['units'],
        dense_units=best_config['dense'],
        dropout_rate=best_config['dropout'],
        learning_rate=best_config['lr'],
        optimizer_name=best_config['opt']
    )

    models_save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")
    os.makedirs(models_save_dir, exist_ok=True)
    checkpoint_path = os.path.join(models_save_dir, "best_lstm_model.keras")

    early_stop = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5, verbose=1)
    checkpoint = ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', save_best_only=True, verbose=0)

    with Timer() as t_train:
        final_hist = final_model.fit(
            X_train_seq, y_train_seq,
            validation_data=(X_val_seq, y_val_seq),
            epochs=15,
            batch_size=best_config['batch_size'],
            class_weight=class_weights,
            callbacks=[early_stop, reduce_lr, checkpoint],
            verbose=1
        )
    training_time = t_train.duration
    print(f"[LSTM TRAINING] Training completed in {training_time:.2f}s. Saved checkpoint to {checkpoint_path}")

    # Final Evaluation on Untouched Test Set
    print("\n[LSTM FINAL EVALUATION] Evaluating on untouched test set...")
    with Timer() as t_infer:
        test_probs = final_model.predict(X_test_seq, batch_size=256, verbose=0)
    inference_time = t_infer.duration
    test_preds = np.argmax(test_probs, axis=1)

    test_metrics, cm = calculate_comprehensive_metrics(y_test_seq, test_preds, test_probs, class_names)
    test_metrics['Training_Time'] = float(training_time)
    test_metrics['Inference_Time'] = float(inference_time)
    test_metrics['Model'] = 'LSTM'

    roc_str = f"{test_metrics['ROC_AUC']:.4f}" if pd.notna(test_metrics['ROC_AUC']) else "N/A"
    pr_str = f"{test_metrics['PR_AUC']:.4f}" if pd.notna(test_metrics['PR_AUC']) else "N/A"
    # Compute Training, Validation, and 5-Fold Cross-Validation Accuracies
    print("[LSTM ACCURACY EVALUATION] Calculating training and validation accuracies...")
    train_probs = final_model.predict(X_train_seq, batch_size=256, verbose=0)
    train_preds = np.argmax(train_probs, axis=1)
    train_accuracy = float(accuracy_score(y_train_seq, train_preds))

    val_probs = final_model.predict(X_val_seq, batch_size=256, verbose=0)
    val_preds = np.argmax(val_probs, axis=1)
    val_accuracy = float(accuracy_score(y_val_seq, val_preds))

    print("[LSTM 5-FOLD CV] Running 5-fold stratified cross-validation on sequence windows...")
    cv_mean, cv_std, _ = perform_5_fold_cv(
        model_or_builder=create_lstm_model,
        X=X_train_seq,
        y=y_train_seq,
        is_keras=True,
        model_kwargs={
            'seq_length': seq_length,
            'num_features': num_features,
            'num_classes': num_classes,
            'lstm_units': best_config['units'],
            'dense_units': best_config['dense'],
            'dropout_rate': best_config['dropout'],
            'learning_rate': best_config['lr'],
            'optimizer_name': best_config['opt']
        },
        fit_kwargs={
            'epochs': 6,
            'batch_size': best_config['batch_size'],
            'class_weight': class_weights
        },
        model_name="LSTM"
    )

    test_metrics['Train_Accuracy'] = train_accuracy
    test_metrics['Val_Accuracy'] = val_accuracy
    test_metrics['CV_Accuracy'] = cv_mean
    test_metrics['CV_Std'] = cv_std
    test_metrics['Test_Accuracy'] = test_metrics['Accuracy']

    print(f"[LSTM RESULTS] Test Accuracy: {test_metrics['Accuracy']:.4f} | Train Accuracy: {train_accuracy:.4f} | Val Accuracy: {val_accuracy:.4f} | 5-Fold CV Acc: {cv_mean:.4f} (±{cv_std:.4f})")
    print(f"[LSTM METRICS] Precision: {test_metrics['Precision']:.4f} | Recall: {test_metrics['Recall']:.4f} | F1: {test_metrics['F1_Score']:.4f}")
    print(f"               ROC-AUC: {roc_str} | PR-AUC: {pr_str}")

    # Save metrics CSV
    df_metrics = pd.DataFrame([test_metrics])
    metrics_path = os.path.join(output_dir, 'metrics', 'lstm_metrics.csv')
    df_metrics.to_csv(metrics_path, index=False)
    print(f"[LSTM] Saved updated metrics to {metrics_path}")

    # Save confusion matrix
    cm_png_path = os.path.join(output_dir, 'confusion_matrices', 'lstm_confusion_matrix.png')
    cm_csv_path = os.path.join(output_dir, 'confusion_matrices', 'lstm_confusion_matrix.csv')
    plot_and_save_confusion_matrix(cm, class_names, cm_png_path, cm_csv_path, model_name="LSTM")

    # Save training curves
    acc_curve_path = os.path.join(output_dir, 'training_curves', 'lstm_accuracy.png')
    loss_curve_path = os.path.join(output_dir, 'training_curves', 'lstm_loss.png')
    plot_and_save_training_curves(final_hist, acc_curve_path, loss_curve_path, model_name="LSTM")

    # Save complete prediction probabilities matrix
    preds_csv_path = os.path.join(output_dir, 'predictions', 'lstm_predictions.csv')
    save_complete_predictions(y_test_seq, test_preds, test_probs, class_names, preds_csv_path, model_name="LSTM")

    # Plot & Save One-vs-Rest ROC Curves
    roc_png_path = os.path.join(output_dir, 'model_comparison', 'lstm_roc_curve.png')
    plot_and_save_roc_curves(y_test_seq, test_probs, class_names, roc_png_path, model_name="LSTM")

    return test_metrics, best_config