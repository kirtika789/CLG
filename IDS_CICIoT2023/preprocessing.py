import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.utils.class_weight import compute_class_weight

DATASET_RAW_DIR = os.environ.get(
    "CICIOT2023_DATA_DIR",
    os.path.expanduser(r"~/.cache/kagglehub/datasets/himadri07/ciciot2023/versions/1/CICIOT23")
)

def find_dataset_files(data_dir=None):
    """
    Locates train.csv, validation.csv, and test.csv across multiple candidate directories:
    1. Explicit data_dir parameter
    2. CICIOT2023_DATA_DIR environment variable
    3. Project local data/raw or data/ directory
    4. Current user's kagglehub cache (~/.cache/kagglehub/datasets/himadri07/ciciot2023)
    5. Fallback path (C:\\Users\\kalas\\.cache\\kagglehub\\...)
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = []
    if data_dir:
        candidates.append(data_dir)
    if 'CICIOT2023_DATA_DIR' in os.environ:
        candidates.append(os.environ['CICIOT2023_DATA_DIR'])
    candidates.append(os.path.join(current_dir, "data", "raw"))
    candidates.append(os.path.join(current_dir, "data"))
    candidates.append(os.path.expanduser(r"~/.cache/kagglehub/datasets/himadri07/ciciot2023/versions/1/CICIOT23"))
    candidates.append(os.path.expanduser(r"~/.cache/kagglehub/datasets/himadri07/ciciot2023"))
    candidates.append(r"C:\Users\kalas\.cache\kagglehub\datasets\himadri07\ciciot2023\versions\1\CICIOT23")

    searched_paths = []
    for cand in candidates:
        if not cand or not os.path.exists(cand):
            searched_paths.append(str(cand) + " (does not exist)")
            continue
        searched_paths.append(str(cand) + " (checked)")
        files = {}
        for root, dirs, filenames in os.walk(cand):
            for f in filenames:
                f_lower = f.lower()
                if f_lower == 'train.csv':
                    files['train'] = os.path.join(root, f)
                elif f_lower == 'validation.csv':
                    files['val'] = os.path.join(root, f)
                elif f_lower == 'test.csv':
                    files['test'] = os.path.join(root, f)
        if files.get('train') and files.get('val') and files.get('test'):
            return files

    searched_summary = "\n  - ".join(searched_paths)
    raise FileNotFoundError(
        f"Could not locate train.csv, validation.csv, and test.csv in any candidate location:\n  - {searched_summary}\n"
        f"Please place train.csv, validation.csv, and test.csv into: {os.path.join(current_dir, 'data', 'raw')}"
    )

def clean_dataframe(df):
    """
    Cleans missing, infinite values, and duplicate records.
    """
    df = df.replace([np.inf, -np.inf], np.nan)
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    df = df.dropna()
    df = df.drop_duplicates()
    return df

def load_file_sampled(filepath, target_samples, random_state=42):
    """
    Efficiently loads a target number of samples from large CSV using chunking.
    """
    chunk_list = []
    total_loaded = 0
    chunk_size = 200000
    for chunk in pd.read_csv(filepath, chunksize=chunk_size):
        chunk_clean = clean_dataframe(chunk)
        chunk_list.append(chunk_clean)
        total_loaded += len(chunk_clean)
        if total_loaded >= target_samples * 2:
            break
    df_combined = pd.concat(chunk_list, ignore_index=True)
    if len(df_combined) > target_samples:
        frac = target_samples / len(df_combined)
        sampled = df_combined.groupby('label').sample(frac=min(1.0, frac), random_state=random_state)
        if len(sampled) > target_samples:
            sampled = sampled.sample(target_samples, random_state=random_state)
        return sampled.reset_index(drop=True)
    return df_combined

def load_and_preprocess_dataset(sample_subset=True, sample_size=200000, random_state=42):
    """
    Loads CICIoT2023 dataset (train, validation, test), cleans data,
    encodes labels, scales features (fit on train ONLY), and returns train/val/test splits.
    """
    files = find_dataset_files()
    print(f"[DATA PREPROCESSING] Dataset files found:")
    print(f"  Train: {files['train']}")
    print(f"  Validation: {files['val']}")
    print(f"  Test: {files['test']}", flush=True)

    total_available_rows = 7845673
    total_available_train = 5491971
    total_available_val = 1176851
    total_available_test = 1176851

    sampling_strategy_desc = "Full Dataset"
    if sample_subset:
        n_train = int(sample_size * 0.70)
        n_val = int(sample_size * 0.15)
        n_test = sample_size - n_train - n_val
        sampling_strategy_desc = f"Stratified Random Subset ({sample_size:,} total samples preserving 70/15/15 ratio)"
        print(f"[DATA PREPROCESSING] Fast-loading sampled subset ({n_train:,} Train, {n_val:,} Val, {n_test:,} Test)...", flush=True)

        df_train = load_file_sampled(files['train'], n_train, random_state=random_state)
        df_val = load_file_sampled(files['val'], n_val, random_state=random_state)
        df_test = load_file_sampled(files['test'], n_test, random_state=random_state)
    else:
        df_train = clean_dataframe(pd.read_csv(files['train']))
        df_val = clean_dataframe(pd.read_csv(files['val']))
        df_test = clean_dataframe(pd.read_csv(files['test']))

    print(f"[DATA PREPROCESSING] Loaded splits — Train: {len(df_train):,}, Val: {len(df_val):,}, Test: {len(df_test):,}", flush=True)

    target_col = 'label'
    feature_cols = [c for c in df_train.columns if c != target_col]

    X_train_raw = df_train[feature_cols].values
    y_train_raw = df_train[target_col].values

    X_val_raw = df_val[feature_cols].values
    y_val_raw = df_val[target_col].values

    X_test_raw = df_test[feature_cols].values
    y_test_raw = df_test[target_col].values

    # Fit LabelEncoder ONLY on training target
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_raw)
    
    known_classes = set(label_encoder.classes_)
    def transform_labels(y_raw):
        y_trans = []
        for l in y_raw:
            if l in known_classes:
                y_trans.append(label_encoder.transform([l])[0])
            else:
                y_trans.append(0)
        return np.array(y_trans)

    y_val = transform_labels(y_val_raw)
    y_test = transform_labels(y_test_raw)

    # Fit StandardScaler ONLY on training features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)

    # Compute class weights to address class imbalance
    unique_classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=unique_classes, y=y_train)
    class_weight_dict = dict(zip(unique_classes, weights))

    dataset_dict = {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'class_weight_dict': class_weight_dict,
        'feature_names': feature_cols,
        'class_names': label_encoder.classes_.tolist(),
        'num_features': len(feature_cols),
        'num_classes': len(label_encoder.classes_),
        'metadata': {
            'total_available_rows': total_available_rows,
            'total_available_train': total_available_train,
            'total_available_val': total_available_val,
            'total_available_test': total_available_test,
            'sample_subset': sample_subset,
            'sampling_strategy': sampling_strategy_desc,
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test),
        }
    }
    return dataset_dict

def apply_pca_preprocessing(X_train, X_val, X_test, n_components=0.95):
    """
    Fits PCA ONLY on training data X_train and transforms X_train, X_val, and X_test.
    """
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)
    X_test_pca = pca.transform(X_test)

    explained_var = pca.explained_variance_ratio_
    cum_var = np.cumsum(explained_var)
    n_selected = pca.n_components_

    pca_stats = {
        'original_features': X_train.shape[1],
        'n_components': n_selected,
        'explained_variance_ratio': explained_var,
        'cumulative_explained_variance': cum_var,
        'total_explained_variance': float(np.sum(explained_var))
    }

    return X_train_pca, X_val_pca, X_test_pca, pca, pca_stats

def sort_by_label(X, y):
    """
    Sorts feature matrix X and label vector y according to target labels.
    Groups same class samples contiguously prior to sequence windowing.
    """
    sort_idx = np.argsort(y)
    return X[sort_idx], y[sort_idx]

def create_sequences(X, y, seq_length=10):
    """
    Converts 2D tabular features (N, F) into 3D sequential sequences (N_seq, seq_length, F)
    for RNN and LSTM models within a split (preventing leakage across split boundaries).
    """
    num_samples = len(X)
    if num_samples < seq_length:
        seq_length = max(1, num_samples)
    
    num_seq = num_samples - seq_length + 1
    num_features = X.shape[1]

    X_seq = np.zeros((num_seq, seq_length, num_features), dtype=np.float32)
    y_seq = np.zeros(num_seq, dtype=np.int32)

    for i in range(num_seq):
        X_seq[i] = X[i : i + seq_length]
        y_seq[i] = y[i + seq_length - 1]

    return X_seq, y_seq

def save_preprocessing_summary(dataset_dict, output_path):
    """
    Saves preprocessing summary CSV file to output/processed_data/preprocessing_summary.csv.
    """
    meta = dataset_dict['metadata']
    summary_data = [{
        'Original Rows': meta['total_available_rows'],
        'Final Rows': meta['train_samples'] + meta['val_samples'] + meta['test_samples'],
        'Original Features': dataset_dict['num_features'],
        'Final Features': dataset_dict['num_features'],
        'Number of Classes': dataset_dict['num_classes'],
        'Training Samples': meta['train_samples'],
        'Validation Samples': meta['val_samples'],
        'Testing Samples': meta['test_samples'],
        'Scaling Method': 'StandardScaler (fit on train ONLY)',
        'Encoding Method': 'LabelEncoder (fit on train ONLY)',
        'PCA Used': 'No (Base Preprocessing)',
        'PCA Components': 'N/A',
        'Class Balancing Method': 'Balanced Class Weights (fit on train ONLY)'
    }]
    df_summary = pd.DataFrame(summary_data)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_summary.to_csv(output_path, index=False)
    print(f"[PREPROCESSING SUMMARY] Saved summary to {output_path}", flush=True)
