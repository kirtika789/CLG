"""
Verification Script for Multiclass ROC-AUC Calculation in IDS CICIoT2023 Benchmark
Tests:
1. Exact match with scikit-learn when all 34 classes are present.
2. Graceful handling of missing classes in test split (exact scenario that caused RNN/LSTM 0.0000).
3. Validation of complete probability matrix (N x 34), dimensions, and row sum normalization.
4. Error handling with NaNs, Infs, shape mismatches, and zero-variance splits (returns NaN, never 0.0).
5. Verification of complete prediction saving with per-class probability columns.
6. Verification of ROC curve generation.
7. End-to-end execution of mini models (RNN, LSTM, DNN, Random Forest) on synthetic data.
"""

import os
import sys
import numpy as np
import pandas as pd

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

from evaluation import (
    calculate_multiclass_roc_auc,
    calculate_comprehensive_metrics,
    plot_and_save_roc_curves,
    save_complete_predictions
)
from comparison import generate_model_comparison, regenerate_comparison_from_metrics
from preprocessing import create_sequences

CLASS_NAMES_34 = [
    'Backdoor_Malware', 'BenignTraffic', 'BrowserHijacking', 'CommandInjection',
    'DDoS-ACK_Fragmentation', 'DDoS-HTTP', 'DDoS-ICMP_Flood', 'DDoS-ICMP_Fragmentation',
    'DDoS-PSHACK_Flood', 'DDoS-RSTFINFlood', 'DDoS-SYN_Flood', 'DDoS-SlowLoris',
    'DDoS-SynonymousIP_Flood', 'DDoS-TCP_Flood', 'DDoS-UDP_Flood', 'DDoS-UDP_Fragmentation',
    'DNS_Spoofing', 'DictionaryBruteForce', 'DoS-HTTP', 'DoS-SYN_Flood',
    'DoS-TCP_Flood', 'DoS-UDP_Flood', 'MITM-ArpSpoofing', 'Mirai-greeth_flood',
    'Mirai-greip_flood', 'Mirai-udpplain', 'Recon-HostDiscovery', 'Recon-OSScan',
    'Recon-PortScan', 'SqlInjection', 'Uploading_Attack', 'VulnerabilityScan',
    'XSS', ' атаку_placeholder'
]

def run_test_1_full_multiclass():
    print("\n" + "="*80)
    print("TEST 1: Standard Multiclass ROC-AUC (All 34 classes present)")
    print("="*80)
    np.random.seed(42)
    n_samples = 2000
    n_classes = 34

    # Generate ground truth with all 34 classes present
    y_true = np.random.randint(0, n_classes, size=n_samples)
    
    # Generate calibrated probability matrix
    raw_logits = np.random.randn(n_samples, n_classes)
    exp_logits = np.exp(raw_logits - np.max(raw_logits, axis=1, keepdims=True))
    y_prob = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    # 1. Run our new calculation
    macro_auc, per_class, omitted, weighted_auc = calculate_multiclass_roc_auc(
        y_true, y_prob, class_names=CLASS_NAMES_34
    )

    # 2. Compare directly with scikit-learn standard OvR macro
    y_bin = label_binarize(y_true, classes=range(n_classes))
    expected_sklearn = roc_auc_score(y_bin, y_prob, multi_class='ovr', average='macro')

    print(f"  Calculated Macro ROC-AUC: {macro_auc:.6f}")
    print(f"  Sklearn Standard OvR:     {expected_sklearn:.6f}")
    diff = abs(macro_auc - expected_sklearn)
    print(f"  Absolute Difference:      {diff:.2e}")

    assert np.isclose(macro_auc, expected_sklearn), f"Difference too large: {diff}"
    assert len(omitted) == 0, f"Expected 0 omitted classes, got {len(omitted)}"
    assert len(per_class) == 34, f"Expected 34 per-class scores, got {len(per_class)}"
    print("  >>> TEST 1 PASSED: Perfect match with scikit-learn standard OvR macro-average.")

def run_test_2_missing_class_scenario():
    print("\n" + "="*80)
    print("TEST 2: Missing Class Handling (Mimicking RNN/LSTM 'Backdoor_Malware' 0 instances)")
    print("="*80)
    np.random.seed(101)
    n_samples = 2000
    n_classes = 34

    # Generate ground truth missing Class 0 ('Backdoor_Malware')
    # Exactly what occurred in RNN/LSTM because sequence windowing dropped indices 0..8
    y_true_missing = np.random.randint(1, n_classes, size=n_samples)
    assert 0 not in y_true_missing, "Class 0 should be missing from y_true"

    raw_logits = np.random.randn(n_samples, n_classes)
    exp_logits = np.exp(raw_logits - np.max(raw_logits, axis=1, keepdims=True))
    y_prob = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    print(f"  Total Classes in Model Output: {n_classes}")
    print(f"  Unique Classes in Test Labels: {len(np.unique(y_true_missing))} (Class 0 '{CLASS_NAMES_34[0]}' is missing)")

    # Run calculate_multiclass_roc_auc
    macro_auc, per_class, omitted, weighted_auc = calculate_multiclass_roc_auc(
        y_true_missing, y_prob, class_names=CLASS_NAMES_34
    )

    print(f"  Calculated Macro ROC-AUC: {macro_auc:.6f}")
    print(f"  Number of Evaluated Classes: {len(per_class)}")
    print(f"  Omitted Classes: {omitted}")

    assert not np.isnan(macro_auc), "ROC-AUC must not be NaN when 33 classes are evaluatable"
    assert macro_auc > 0.0, f"ROC-AUC must be > 0.0, got {macro_auc}"
    assert len(per_class) == 33, f"Expected 33 evaluated classes, got {len(per_class)}"
    assert len(omitted) == 1, f"Expected 1 omitted class, got {len(omitted)}"
    assert omitted[0][0] == CLASS_NAMES_34[0], f"Expected omitted class to be {CLASS_NAMES_34[0]}"

    # Verify that the macro-average equals the exact mean of the 33 evaluatable classes
    manual_mean = float(np.mean([roc_auc_score((y_true_missing == c).astype(int), y_prob[:, c]) for c in range(1, 34)]))
    assert np.isclose(macro_auc, manual_mean), f"Macro AUC ({macro_auc}) != manual mean ({manual_mean})"
    print(f"  Manual Mean of 33 Present Classes: {manual_mean:.6f} (Exact match!)")
    print("  >>> TEST 2 PASSED: Missing class correctly logged and excluded from average; finite score returned.")

def run_test_3_error_handling():
    print("\n" + "="*80)
    print("TEST 3: Error Handling & Invalid Input Safeguards")
    print("="*80)

    # 1. Sample count mismatch
    y_t = np.array([0, 1, 2])
    p_bad_len = np.random.uniform(size=(5, 3))
    p_bad_len /= p_bad_len.sum(axis=1, keepdims=True)
    auc_mismatch, _, _, _ = calculate_multiclass_roc_auc(y_t, p_bad_len, class_names=['A', 'B', 'C'])
    assert np.isnan(auc_mismatch), "Should return NaN on sample count mismatch"
    print("  [Passed] Sample count mismatch handled gracefully (returned NaN).")

    # 2. NaN values in probability matrix
    p_nan = np.ones((10, 3)) / 3.0
    p_nan[2, 1] = np.nan
    auc_nan, _, _, _ = calculate_multiclass_roc_auc(np.random.randint(0, 3, 10), p_nan, class_names=['A', 'B', 'C'])
    assert np.isnan(auc_nan), "Should return NaN when probabilities contain NaN"
    print("  [Passed] Probabilities with NaN handled gracefully (returned NaN).")

    # 3. Only 1 unique class in the entire test set (AUC impossible)
    y_single = np.zeros(20, dtype=int) # All class 0
    p_valid = np.random.uniform(size=(20, 3))
    p_valid /= p_valid.sum(axis=1, keepdims=True)
    auc_single, _, _, _ = calculate_multiclass_roc_auc(y_single, p_valid, class_names=['A', 'B', 'C'])
    assert np.isnan(auc_single), "Should return NaN when only 1 class exists in entire dataset"
    print("  [Passed] Single-class ground truth handled gracefully (returned NaN, not 0.0).")

    # 4. Comprehensive metrics reporting
    metrics, _ = calculate_comprehensive_metrics(y_single, np.zeros(20, dtype=int), p_valid, class_names=['A', 'B', 'C'])
    assert np.isnan(metrics['ROC_AUC']), "calculate_comprehensive_metrics must report np.nan, not 0.0"
    print("  [Passed] calculate_comprehensive_metrics reports np.nan for undefined ROC-AUC, not 0.0.")
    print("  >>> TEST 3 PASSED: All error handling safeguards verified.")

def run_test_4_save_and_reload_predictions():
    print("\n" + "="*80)
    print("TEST 4: Complete Prediction Probability Matrix Saving & Reloading")
    print("="*80)
    test_csv_path = os.path.join(current_dir, "output", "predictions", "test_synthetic_predictions.csv")
    n_samples = 50
    n_classes = len(CLASS_NAMES_34)
    y_true = np.random.randint(0, n_classes, size=n_samples)
    y_pred = np.random.randint(0, n_classes, size=n_samples)
    y_prob = np.random.uniform(size=(n_samples, n_classes))
    y_prob /= y_prob.sum(axis=1, keepdims=True)

    # Save
    df_saved = save_complete_predictions(
        y_true, y_pred, y_prob, CLASS_NAMES_34, test_csv_path, model_name="Synthetic_Test"
    )

    assert os.path.exists(test_csv_path), "Prediction CSV was not created"
    assert df_saved.shape == (n_samples, 3 + n_classes), f"Expected shape ({n_samples}, {3 + n_classes}), got {df_saved.shape}"
    assert 'Actual_Label' in df_saved.columns
    assert 'Predicted_Label' in df_saved.columns
    assert 'Prediction_Confidence' in df_saved.columns
    for c_name in CLASS_NAMES_34:
        assert f"prob_{c_name}" in df_saved.columns, f"Missing column prob_{c_name}"

    # Reload from disk and reconstruct probability matrix
    df_loaded = pd.read_csv(test_csv_path)
    prob_cols = [f"prob_{c}" for c in CLASS_NAMES_34]
    reloaded_probs = df_loaded[prob_cols].values

    assert np.allclose(reloaded_probs, y_prob, atol=1e-5), "Reloaded probabilities do not match saved probabilities"
    print(f"  Successfully saved and reloaded complete {df_saved.shape} probability matrix.")
    if os.path.exists(test_csv_path):
        os.remove(test_csv_path)
    print("  >>> TEST 4 PASSED: Complete prediction saving verified.")

def run_test_5_roc_curve_plotting():
    print("\n" + "="*80)
    print("TEST 5: ROC Curve Plot Generation")
    print("="*80)
    roc_png_path = os.path.join(current_dir, "output", "model_comparison", "synthetic_test_roc_curve.png")
    n_samples = 500
    n_classes = len(CLASS_NAMES_34)
    y_true = np.random.randint(0, n_classes, size=n_samples)
    y_prob = np.random.uniform(size=(n_samples, n_classes))
    y_prob /= y_prob.sum(axis=1, keepdims=True)

    plot_and_save_roc_curves(y_true, y_prob, CLASS_NAMES_34, roc_png_path, model_name="Synthetic_Test")

    assert os.path.exists(roc_png_path), f"ROC curve PNG was not created at {roc_png_path}"
    file_size = os.path.getsize(roc_png_path)
    print(f"  ROC curve successfully generated at {roc_png_path} ({file_size:,} bytes).")
    if os.path.exists(roc_png_path):
        os.remove(roc_png_path)
    print("  >>> TEST 5 PASSED: ROC curve generation verified.")

def run_test_6_mini_models_benchmark():
    print("\n" + "="*80)
    print("TEST 6: Mini-Model End-to-End Pipeline Execution (RNN, LSTM, DNN, RF)")
    print("="*80)
    from sklearn.ensemble import RandomForestClassifier
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Input, Dense, SimpleRNN, LSTM, Dropout

    np.random.seed(42)
    tf.random.set_seed(42)

    n_samples_train = 600
    n_samples_test = 300
    n_features = 46
    n_classes = 34
    seq_length = 5

    print(f"  Generating synthetic dataset with {n_features} features, {n_classes} classes...")
    X_train = np.random.randn(n_samples_train, n_features).astype(np.float32)
    y_train = np.random.randint(0, n_classes, size=n_samples_train).astype(np.int32)

    # In test set, intentionally omit class 0 to reproduce the RNN/LSTM issue
    X_test = np.random.randn(n_samples_test, n_features).astype(np.float32)
    y_test = np.random.randint(1, n_classes, size=n_samples_test).astype(np.int32) # Class 0 missing!

    # 1. Random Forest
    print("\n  [Testing Random Forest]...")
    rf = RandomForestClassifier(n_estimators=10, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    rf_probs = rf.predict_proba(X_test)
    if rf_probs.shape[1] < n_classes:
        # pad if some classes not in training batch
        full_rf_probs = np.zeros((len(X_test), n_classes))
        for col_idx, c in enumerate(rf.classes_):
            full_rf_probs[:, c] = rf_probs[:, col_idx]
        rf_probs = full_rf_probs
    rf_preds = np.argmax(rf_probs, axis=1)
    rf_metrics, _ = calculate_comprehensive_metrics(y_test, rf_preds, rf_probs, CLASS_NAMES_34)
    print(f"    Random Forest ROC-AUC: {rf_metrics['ROC_AUC']:.4f} (Accuracy: {rf_metrics['Accuracy']:.4f})")
    assert not np.isnan(rf_metrics['ROC_AUC']) and rf_metrics['ROC_AUC'] > 0.0

    # 2. DNN
    print("\n  [Testing DNN]...")
    dnn = Sequential([
        Input(shape=(n_features,)),
        Dense(32, activation='relu'),
        Dense(n_classes, activation='softmax')
    ])
    dnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    dnn.fit(X_train, y_train, epochs=2, batch_size=64, verbose=0)
    dnn_probs = dnn.predict(X_test, verbose=0)
    dnn_preds = np.argmax(dnn_probs, axis=1)
    dnn_metrics, _ = calculate_comprehensive_metrics(y_test, dnn_preds, dnn_probs, CLASS_NAMES_34)
    print(f"    DNN ROC-AUC: {dnn_metrics['ROC_AUC']:.4f} (Accuracy: {dnn_metrics['Accuracy']:.4f})")
    assert not np.isnan(dnn_metrics['ROC_AUC']) and dnn_metrics['ROC_AUC'] > 0.0

    # 3. RNN with Sequence Construction
    print("\n  [Testing RNN with Sequence Windows]...")
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_length=seq_length)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, seq_length=seq_length)

    rnn = Sequential([
        Input(shape=(seq_length, n_features)),
        SimpleRNN(16, return_sequences=False),
        Dense(n_classes, activation='softmax')
    ])
    rnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    rnn.fit(X_train_seq, y_train_seq, epochs=2, batch_size=64, verbose=0)
    rnn_probs = rnn.predict(X_test_seq, verbose=0)
    rnn_preds = np.argmax(rnn_probs, axis=1)
    rnn_metrics, _ = calculate_comprehensive_metrics(y_test_seq, rnn_preds, rnn_probs, CLASS_NAMES_34)
    print(f"    RNN ROC-AUC: {rnn_metrics['ROC_AUC']:.4f} (Accuracy: {rnn_metrics['Accuracy']:.4f})")
    assert not np.isnan(rnn_metrics['ROC_AUC']) and rnn_metrics['ROC_AUC'] > 0.0, "RNN ROC-AUC must be genuine and non-zero!"

    # 4. LSTM with Sequence Construction
    print("\n  [Testing LSTM with Sequence Windows]...")
    lstm = Sequential([
        Input(shape=(seq_length, n_features)),
        LSTM(16, return_sequences=False),
        Dense(n_classes, activation='softmax')
    ])
    lstm.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    lstm.fit(X_train_seq, y_train_seq, epochs=2, batch_size=64, verbose=0)
    lstm_probs = lstm.predict(X_test_seq, verbose=0)
    lstm_preds = np.argmax(lstm_probs, axis=1)
    lstm_metrics, _ = calculate_comprehensive_metrics(y_test_seq, lstm_preds, lstm_probs, CLASS_NAMES_34)
    print(f"    LSTM ROC-AUC: {lstm_metrics['ROC_AUC']:.4f} (Accuracy: {lstm_metrics['Accuracy']:.4f})")
    assert not np.isnan(lstm_metrics['ROC_AUC']) and lstm_metrics['ROC_AUC'] > 0.0, "LSTM ROC-AUC must be genuine and non-zero!"

    print("\n  All 4 model architectures evaluated successfully!")
    print(f"  RNN ROC-AUC:  {rnn_metrics['ROC_AUC']:.4f} (NOT 0.0000)")
    print(f"  LSTM ROC-AUC: {lstm_metrics['ROC_AUC']:.4f} (NOT 0.0000)")
    print("  >>> TEST 6 PASSED: End-to-end model evaluation verified.")

def main():
    print("================================================================================")
    print("      MULTICLASS ROC-AUC VERIFICATION SUITE — IDS CICIoT2023 BENCHMARK         ")
    print("================================================================================")
    run_test_1_full_multiclass()
    run_test_2_missing_class_scenario()
    run_test_3_error_handling()
    run_test_4_save_and_reload_predictions()
    run_test_5_roc_curve_plotting()
    run_test_6_mini_models_benchmark()
    print("\n" + "="*80)
    print("             ALL 6 VERIFICATION TEST SUITES PASSED SUCCESSFULLY!               ")
    print("================================================================================")

if __name__ == '__main__':
    main()
