import os
import sys
import argparse

# Ensure current project directory is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from utils import set_seeds, ensure_directories
from preprocessing import (
    load_and_preprocess_dataset,
    save_preprocessing_summary
)
from models.dnn import run_dnn_experiment
from models.rnn import run_rnn_experiment
from models.random_forest import run_random_forest_experiment
from models.pca_lnn import run_pca_lnn_experiment
from models.pca_dnn import run_pca_dnn_experiment
from models.lstm import run_lstm_experiment
from comparison import generate_model_comparison

def main():
    parser = argparse.ArgumentParser(description="IDS CICIoT2023 Machine Learning & Deep Learning Experiment Pipeline")
    parser.add_argument('--full-dataset', action='store_true', help="Use complete 7.85 million dataset instead of stratified experimental subset")
    parser.add_argument('--sample-size', type=int, default=200000, help="Number of records for experimental subset (default: 200,000)")
    args = parser.parse_args()

    print("==========================================================================================")
    print("      INTRUSION DETECTION SYSTEM (IDS) — CICIOT2023 BENCHMARK EXPERIMENT PIPELINE         ")
    print("==========================================================================================")

    # 1. Reproducibility & Output Directories
    set_seeds(42)
    ensure_directories(current_dir)
    output_dir = os.path.join(current_dir, "output")

    # 2. Data Loading & Preprocessing
    sample_subset = not args.full_dataset
    print(f"\n[PIPELINE START] Loading CICIoT2023 dataset (Subset Mode: {sample_subset}, Sample Size: {args.sample_size:,})...")
    
    dataset_dict = load_and_preprocess_dataset(
        sample_subset=sample_subset,
        sample_size=args.sample_size,
        random_state=42
    )

    # 3. Save Preprocessing Summary CSV
    prep_summary_path = os.path.join(output_dir, "processed_data", "preprocessing_summary.csv")
    save_preprocessing_summary(dataset_dict, prep_summary_path)

    # 4. Model Experiments
    all_metrics = []

    # Model 1: DNN
    m1_metrics, m1_cfg = run_dnn_experiment(dataset_dict, output_dir)
    all_metrics.append(m1_metrics)

    # Model 2: RNN
    m2_metrics, m2_cfg = run_rnn_experiment(dataset_dict, output_dir, seq_length=10)
    all_metrics.append(m2_metrics)

    # Model 3: Random Forest
    m3_metrics, m3_cfg = run_random_forest_experiment(dataset_dict, output_dir)
    all_metrics.append(m3_metrics)

    # Model 4: PCA + LNN
    m4_metrics, m4_cfg = run_pca_lnn_experiment(dataset_dict, output_dir)
    all_metrics.append(m4_metrics)

    # Model 5: PCA + DNN
    m5_metrics, m5_cfg = run_pca_dnn_experiment(dataset_dict, output_dir)
    all_metrics.append(m5_metrics)

    # Model 6: LSTM
    m6_metrics, m6_cfg = run_lstm_experiment(dataset_dict, output_dir, seq_length=10)
    all_metrics.append(m6_metrics)

    # 5. Consolidation, Comparison & Academic Report Generation
    print("\n==================== CONSOLIDATING RESULTS & GENERATING REPORT ====================")
    df_comp = generate_model_comparison(all_metrics, output_dir, dataset_dict['metadata'])

    # 6. Final Concise Text Summary Output
    print("\n" + "="*90)
    print("                                FINAL EXPERIMENTAL SUMMARY                                ")
    print("="*90)
    print(f"Dataset Size: {dataset_dict['metadata']['train_samples'] + dataset_dict['metadata']['val_samples'] + dataset_dict['metadata']['test_samples']:,} samples ({dataset_dict['metadata']['sampling_strategy']})")
    print(f"Features Count: {dataset_dict['num_features']} | Classes Count: {dataset_dict['num_classes']}")
    print("-" * 90)

    for m in all_metrics:
        print(f"[{m['Model']}]")
        print(f"  Accuracy : {m['Accuracy']:.4f}")
        print(f"  Precision: {m['Precision']:.4f}")
        print(f"  Recall   : {m['Recall']:.4f}")
        print(f"  F1-Score : {m['F1_Score']:.4f}")
        print(f"  FPR      : {m['FPR']:.4f} | FNR: {m['FNR']:.4f}")
        print(f"  Train Time: {m['Training_Time']:.2f}s | Infer Time: {m['Inference_Time']:.2f}s")
        print("-" * 90)

    best_model_row = df_comp.loc[df_comp['F1_Score'].idxmax()]
    print(f"BEST PERFORMING MODEL (by F1-Score): {best_model_row['Model']}")
    print(f"  Accuracy: {best_model_row['Accuracy']:.4f} | F1: {best_model_row['F1_Score']:.4f}")
    print(f"All generated metrics, plots, and report saved in: {output_dir}")
    print("="*90)

if __name__ == '__main__':
    main()
