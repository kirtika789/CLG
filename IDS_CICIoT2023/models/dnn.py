import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

# Assuming these are your local custom modules
from utils import set_seeds, Timer
from evaluation import (
    calculate_comprehensive_metrics,
    plot_and_save_confusion_matrix,
    plot_and_save_training_curves
)

def create_dnn_model(input_dim, num_classes, layers_config=[256, 128], dropout_rate=0.2, learning_rate=0.001, optimizer_name='adam'):
    """
    Constructs a Deep Neural Network model with L2 regularization and Swish activation.
    """
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    
    # First Hidden Layer
    model.add(Dense(layers_config[0], activation='swish', kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    # Subsequent Hidden Layers
    for units in layers_config[1:]:
        model.add(Dense(units, activation='swish', kernel_regularizer=l2(1e-4)))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))

    # Output Layer
    model.add(Dense(num_classes, activation='softmax'))

    # Optimizer selection
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

def run_dnn_experiment(dataset_dict, output_dir):
    """
    Runs hyperparameter tuning, model training, evaluation, and saving for DNN.
    """
    set_seeds(42)
    print("\n==================== MODEL 1: DEEP NEURAL NETWORK (DNN) ====================")

    X_train, y_train = dataset_dict['X_train'], dataset_dict['y_train']
    X_val, y_val = dataset_dict['X_val'], dataset_dict['y_val']
    X_test, y_test = dataset_dict['X_test'], dataset_dict['y_test']
    class_names = dataset_dict['class_names']
    input_dim = dataset_dict['num_features']
    num_classes = dataset_dict['num_classes']
    class_weights = dataset_dict['class_weight_dict']

    # Expanded Search space
    hyperparam_configs = [
        # Original configs
        {'layers': [256, 128], 'lr': 0.001, 'batch_size': 128, 'opt': 'adam', 'dropout': 0.2},
        {'layers': [512, 256, 128], 'lr': 0.001, 'batch_size': 128, 'opt': 'adam', 'dropout': 0.3},
        {'layers': [128, 64], 'lr': 0.0005, 'batch_size': 64, 'opt': 'rmsprop', 'dropout': 0.2},
        {'layers': [256, 128, 64], 'lr': 0.0001, 'batch_size': 128, 'opt': 'adam', 'dropout': 0.2},
        
        # New configs for better generalization
        {'layers': [256, 128], 'lr': 0.001, 'batch_size': 32, 'opt': 'adam', 'dropout': 0.2}, # Smaller batch size
        {'layers': [512, 512, 256], 'lr': 0.001, 'batch_size': 64, 'opt': 'adam', 'dropout': 0.4}, # Wider/Deeper + high dropout
        {'layers': [128], 'lr': 0.001, 'batch_size': 64, 'opt': 'adam', 'dropout': 0.1}, # Shallower network
    ]

    tuning_results = []
    best_val_f1 = -1.0
    best_config = None
    best_history = None
    best_model = None

    print(f"[DNN TUNING] Testing {len(hyperparam_configs)} hyperparameter configurations on training/validation split...")

    for i, cfg in enumerate(hyperparam_configs):
        print(f"  Config {i+1}/{len(hyperparam_configs)}: Layers={cfg['layers']}, Opt={cfg['opt']}, LR={cfg['lr']}, Batch={cfg['batch_size']}, Dropout={cfg['dropout']}")
        model = create_dnn_model(
            input_dim=input_dim,
            num_classes=num_classes,
            layers_config=cfg['layers'],
            dropout_rate=cfg['dropout'],
            learning_rate=cfg['lr'],
            optimizer_name=cfg['opt']
        )

        # Updated callbacks for tuning
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=0)
        
        hist = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100, # Increased max epochs; EarlyStopping will handle halting
            batch_size=cfg['batch_size'],
            class_weight=class_weights,
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )

        val_probs = model.predict(X_val, batch_size=256, verbose=0)
        val_preds = np.argmax(val_probs, axis=1)

        val_metrics, _ = calculate_comprehensive_metrics(y_val, val_preds, val_probs, class_names)

        tuning_results.append({
            'Model': 'DNN',
            'Optimizer': cfg['opt'],
            'Learning Rate': cfg['lr'],
            'Batch Size': cfg['batch_size'],
            'Layers': str(cfg['layers']),
            'Neurons': str(cfg['layers']),
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
            best_history = hist.history
            best_model = model

    # Save hyperparameter tuning results
    os.makedirs(os.path.join(output_dir, 'hyperparameter_tuning'), exist_ok=True)
    df_tuning = pd.DataFrame(tuning_results)
    tuning_csv_path = os.path.join(output_dir, 'hyperparameter_tuning', 'dnn_tuning_results.csv')
    df_tuning.to_csv(tuning_csv_path, index=False)
    print(f"[DNN TUNING] Saved all tuning results to {tuning_csv_path}")
    print(f"[DNN TUNING] Winning Config: Layers={best_config['layers']}, Opt={best_config['opt']}, LR={best_config['lr']} (Val F1: {best_val_f1:.4f})")

    # Retrain winning model cleanly to record exact timing & training history curves
    print("\n[DNN FINAL TRAINING] Retraining best configuration...")
    final_model = create_dnn_model(
        input_dim=input_dim,
        num_classes=num_classes,
        layers_config=best_config['layers'],
        dropout_rate=best_config['dropout'],
        learning_rate=best_config['lr'],
        optimizer_name=best_config['opt']
    )
    
    # Extended patience and dynamic learning rate for the final deep training phase
    early_stop_final = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    reduce_lr_final = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1)

    with Timer() as t_train:
        final_hist = final_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=150, # Greatly increased max epochs
            batch_size=best_config['batch_size'],
            class_weight=class_weights,
            callbacks=[early_stop_final, reduce_lr_final],
            verbose=1
        )
    training_time = t_train.duration
    print(f"[DNN TRAINING] Training completed in {training_time:.2f} seconds.")

    # Final Test Set Evaluation
    print("[DNN FINAL EVALUATION] Evaluating on untouched test set...")
    with Timer() as t_infer:
        test_probs = final_model.predict(X_test, batch_size=256, verbose=0)
    inference_time = t_infer.duration
    test_preds = np.argmax(test_probs, axis=1)

    test_metrics, cm = calculate_comprehensive_metrics(y_test, test_preds, test_probs, class_names)
    test_metrics['Training_Time'] = float(training_time)
    test_metrics['Inference_Time'] = float(inference_time)
    test_metrics['Model'] = 'DNN'

    print(f"[DNN RESULTS] Test Accuracy: {test_metrics['Accuracy']:.4f} | Precision: {test_metrics['Precision']:.4f} | Recall: {test_metrics['Recall']:.4f} | F1: {test_metrics['F1_Score']:.4f}")

    # Ensure directories exist and save files
    os.makedirs(os.path.join(output_dir, 'metrics'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'confusion_matrices'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'training_curves'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'predictions'), exist_ok=True)

    # Save Metrics CSV
    df_metrics = pd.DataFrame([test_metrics])
    metrics_csv_path = os.path.join(output_dir, 'metrics', 'dnn_metrics.csv')
    df_metrics.to_csv(metrics_csv_path, index=False)

    # Save Confusion Matrix
    cm_png_path = os.path.join(output_dir, 'confusion_matrices', 'dnn_confusion_matrix.png')
    cm_csv_path = os.path.join(output_dir, 'confusion_matrices', 'dnn_confusion_matrix.csv')
    plot_and_save_confusion_matrix(cm, class_names, cm_png_path, cm_csv_path, model_name="DNN")

    # Save Training Curves
    acc_curve_path = os.path.join(output_dir, 'training_curves', 'dnn_accuracy.png')
    loss_curve_path = os.path.join(output_dir, 'training_curves', 'dnn_loss.png')
    plot_and_save_training_curves(final_hist, acc_curve_path, loss_curve_path, model_name="DNN")

    # Save Predictions CSV
    y_test_labels = [class_names[i] for i in y_test]
    pred_labels = [class_names[i] for i in test_preds]
    max_probs = np.max(test_probs, axis=1)

    df_preds = pd.DataFrame({
        'Actual_Label': y_test_labels,
        'Predicted_Label': pred_labels,
        'Prediction_Probability': max_probs
    })
    preds_csv_path = os.path.join(output_dir, 'predictions', 'dnn_predictions.csv')
    df_preds.to_csv(preds_csv_path, index=False)
    print(f"[DNN PREDICTIONS] Saved test predictions to {preds_csv_path}")

    return test_metrics, best_config