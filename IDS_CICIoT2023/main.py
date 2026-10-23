import os
import sys
import argparse
import pandas as pd
# pyrefly: ignore[missing-import]
import numpy as np

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
from comparison import generate_model_comparison, regenerate_comparison_from_metrics

def main():
    parser = argparse.ArgumentParser(description="IDS CICIoT2023 Machine Learning & Deep Learning Experiment Pipeline")
    parser.add_argument('--full-dataset', action='store_true', help="Use complete 7.85 million dataset instead of stratified experimental subset")
    parser.add_argument('--sample-size', type=int, default=200000, help="Number of records for experimental subset (default: 200,000)")
    parser.add_argument('--model', type=str, default='all', choices=['all', 'dnn', 'rnn', 'rf', 'random_forest', 'pca_lnn', 'pca_dnn', 'lstm'], help="Specific model to run, or 'all' (default: all)")
    parser.add_argument('--report-only', action='store_true', help="Regenerate comparison charts and markdown report from existing metrics without retraining")
    args = parser.parse_args()

    print("==========================================================================================")
    print("      INTRUSION DETECTION SYSTEM (IDS) — CICIOT2023 BENCHMARK EXPERIMENT PIPELINE         ")
    print("==========================================================================================")

    # 1. Reproducibility & Output Directories
    set_seeds(42)
    ensure_directories(current_dir)
    output_dir = os.path.join(current_dir, "output")

    # If --report-only, regenerate directly and exit
    if args.report_only:
        print("\n[REPORT MODE] Regenerating comparison charts and academic report from saved metrics...")
        df_comp = regenerate_comparison_from_metrics(output_dir)
        if df_comp is not None:
            print("\n" + "="*90)
            print("                                UPDATED MODEL COMPARISON                                  ")
            print("="*90)
            print(df_comp[['Model', 'Accuracy', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC', 'Training_Time', 'Inference_Time']].to_string(index=False))
            print("="*90)
        return

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
    model_choice = args.model.lower()
    run_all = (model_choice == 'all')
    all_metrics = []

    # Model 1: DNN
    if run_all or model_choice == 'dnn':
        m1_metrics, m1_cfg = run_dnn_experiment(dataset_dict, output_dir)
        all_metrics.append(m1_metrics)

    # Model 2: RNN
    if run_all or model_choice == 'rnn':
        m2_metrics, m2_cfg = run_rnn_experiment(dataset_dict, output_dir, seq_length=10)
        all_metrics.append(m2_metrics)

    # Model 3: Random Forest
    if run_all or model_choice in ['rf', 'random_forest']:
        m3_metrics, m3_cfg = run_random_forest_experiment(dataset_dict, output_dir)
        all_metrics.append(m3_metrics)

    # Model 4: PCA + LNN
    if run_all or model_choice == 'pca_lnn':
        m4_metrics, m4_cfg = run_pca_lnn_experiment(dataset_dict, output_dir)
        all_metrics.append(m4_metrics)

    # Model 5: PCA + DNN
    if run_all or model_choice == 'pca_dnn':
        m5_metrics, m5_cfg = run_pca_dnn_experiment(dataset_dict, output_dir)
        all_metrics.append(m5_metrics)

    # Model 6: LSTM
    if run_all or model_choice == 'lstm':
        m6_metrics, m6_cfg = run_lstm_experiment(dataset_dict, output_dir, seq_length=10)
        all_metrics.append(m6_metrics)

    # If running a subset of models, load cached metrics for the rest to build a complete report
    if not run_all:
        metrics_dir = os.path.join(output_dir, 'metrics')
        model_file_map = {
            'DNN': 'dnn_metrics.csv',
            'RNN': 'rnn_metrics.csv',
            'Random Forest': 'random_forest_metrics.csv',
            'PCA + LNN': 'pca_lnn_metrics.csv',
            'PCA + DNN': 'pca_dnn_metrics.csv',
            'LSTM': 'lstm_metrics.csv'
        }
        already_run = {m['Model'] for m in all_metrics}
        for mod_name, fname in model_file_map.items():
            if mod_name not in already_run:
                fpath = os.path.join(metrics_dir, fname)
                if os.path.exists(fpath):
                    df_m = pd.read_csv(fpath)
                    if len(df_m) > 0:
                        all_metrics.append(df_m.iloc[0].to_dict())

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
        roc_disp = f"{m['ROC_AUC']:.4f}" if ('ROC_AUC' in m and pd.notna(m['ROC_AUC'])) else "N/A"
        pr_disp = f"{m['PR_AUC']:.4f}" if ('PR_AUC' in m and pd.notna(m['PR_AUC'])) else "N/A"
        print(f"[{m['Model']}]")
        print(f"  Accuracy : {m['Accuracy']:.4f}")
        print(f"  Precision: {m['Precision']:.4f}")
        print(f"  Recall   : {m['Recall']:.4f}")
        print(f"  F1-Score : {m['F1_Score']:.4f}")
        print(f"  ROC-AUC  : {roc_disp} | PR-AUC: {pr_disp}")
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
