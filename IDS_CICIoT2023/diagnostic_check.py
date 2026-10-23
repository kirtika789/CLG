import os
import sys
# pyrefly: ignore[missing-import]
import numpy as np
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from preprocessing import load_and_preprocess_dataset, create_sequences
from models.rnn import create_rnn_model
from models.lstm import create_lstm_model
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from evaluation import calculate_multiclass_roc_auc

print("="*75)
print("DIAGNOSTIC REPORT: DATASET, LABELS, SEQUENCES, AND METRICS")
print("="*75)

d = load_and_preprocess_dataset(sample_subset=True, sample_size=200000, random_state=42)
y_test = d['y_test']
class_names = d['class_names']
n_classes = len(class_names)

print(f"\n1. DATASET TEST LABELS (tabular):")
print(f"   Shape: {y_test.shape}, dtype: {y_test.dtype}")
print(f"   Encoding: Integer label-encoded (multiclass, values 0 to {n_classes-1})")
print(f"   Unique classes in y_test: {len(np.unique(y_test))}/{n_classes}")
test_counts = pd.Series(y_test).value_counts().sort_index()
print("   Least frequent classes in y_test:")
for idx in test_counts.head(5).index:
    print(f"     Class {idx} ({class_names[idx]}): {test_counts[idx]} samples")

X_test_seq, y_test_seq = create_sequences(d['X_test'], y_test, seq_length=10)
print(f"\n2. SEQUENCE TEST LABELS (seq_length=10):")
print(f"   Shape: {y_test_seq.shape}, dtype: {y_test_seq.dtype}")
print(f"   Unique classes in y_test_seq: {len(np.unique(y_test_seq))}/{n_classes}")
seq_counts = pd.Series(y_test_seq).value_counts().sort_index()
missing_classes = set(range(n_classes)) - set(np.unique(y_test_seq))
print(f"   Missing classes in y_test_seq: {[class_names[c] for c in missing_classes]} (indices: {missing_classes})")

print(f"\n3. EXISTING PREDICTION FILES ON DISK:")
for model_name in ['rnn', 'lstm']:
    pred_path = os.path.join(current_dir, 'output', 'predictions', f'{model_name}_predictions.csv')
    df_p = pd.read_csv(pred_path)
    col_name = 'Prediction_Probability' if 'Prediction_Probability' in df_p.columns else 'Prediction_Confidence'
    print(f"   [{model_name.upper()}]:")
    print(f"     File: {pred_path}")
    print(f"     Shape: {df_p.shape}, Columns: {list(df_p.columns)}")
    print(f"     {col_name} dtype: {df_p[col_name].dtype}")
    print(f"     Min: {df_p[col_name].min()}, Max: {df_p[col_name].max()}")
    print(f"     Contains NaN: {df_p[col_name].isna().any()}")
    print(f"     Contains Inf: {np.isinf(df_p[col_name]).any()}")
    print(f"     First 3 samples:\n{df_p.head(3)}")

print(f"\n4. MODEL ARCHITECTURES:")
rnn_m = create_rnn_model(10, 46, 34)
lstm_m = create_lstm_model(10, 46, 34)
print(f"   RNN output layer: {rnn_m.layers[-1].name}, units: {rnn_m.layers[-1].units}, activation: {rnn_m.layers[-1].activation.__name__}")
print(f"   LSTM output layer: {lstm_m.layers[-1].name}, units: {lstm_m.layers[-1].units}, activation: {lstm_m.layers[-1].activation.__name__}")

print(f"\n5. ROC-AUC EXCEPTION TEST ON y_test_seq:")
dummy_probs = np.random.uniform(size=(len(y_test_seq), n_classes))
dummy_probs /= dummy_probs.sum(axis=1, keepdims=True)
y_bin = label_binarize(y_test_seq, classes=range(n_classes))
try:
    roc_auc_score(y_bin, dummy_probs, multi_class='ovr', average='macro')
except Exception as e:
    print(f"   Standard sklearn roc_auc_score exception: {type(e).__name__}: {e}")

macro_auc, per_class, omitted, _ = calculate_multiclass_roc_auc(y_test_seq, dummy_probs, class_names)
print(f"   Robust calculate_multiclass_roc_auc result: {macro_auc:.4f}, omitted classes count: {len(omitted)}")
print(f"   Omitted class detail: {omitted}")
