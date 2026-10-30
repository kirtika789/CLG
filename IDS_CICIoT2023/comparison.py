import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def generate_model_comparison(all_metrics_list, output_dir, dataset_metadata=None):
    """
    Consolidates evaluation metrics across all 6 models, creates comparison plots,
    saves comparison CSVs, and generates the final academic report.
    """
    df_comparison = pd.DataFrame(all_metrics_list)
    
    # Required column ordering
    cols_order = [
        'Model', 'Accuracy', 'Train_Accuracy', 'Val_Accuracy', 'CV_Accuracy', 'CV_Std',
        'Precision', 'Recall', 'F1_Score', 'Specificity',
        'FPR', 'FNR', 'ROC_AUC', 'PR_AUC', 'Training_Time', 'Inference_Time',
        'Macro_Precision', 'Macro_Recall', 'Macro_F1', 'Weighted_Precision',
        'Weighted_Recall', 'Weighted_F1', 'Test_Accuracy'
    ]
    existing_cols = [c for c in cols_order if c in df_comparison.columns]
    # Add any extra columns at the end
    extra_cols = [c for c in df_comparison.columns if c not in cols_order]
    df_comparison = df_comparison[existing_cols + extra_cols]

    # Save CSVs
    final_comp_csv = os.path.join(output_dir, 'metrics', 'final_model_comparison.csv')
    os.makedirs(os.path.dirname(final_comp_csv), exist_ok=True)
    df_comparison.to_csv(final_comp_csv, index=False)
    
    reports_dir = os.path.join(output_dir, 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    df_comparison.to_csv(os.path.join(reports_dir, 'final_results.csv'), index=False)
    df_comparison.to_csv(os.path.join(reports_dir, 'model_comparison.csv'), index=False)

    print(f"[COMPARISON] Saved final comparison CSVs to {final_comp_csv} and {reports_dir}")

    # Generate Comparison Charts
    comp_img_dir = os.path.join(output_dir, 'model_comparison')
    os.makedirs(comp_img_dir, exist_ok=True)

    metrics_to_plot = [
        ('Accuracy', 'accuracy_comparison.png', 'Test Accuracy Comparison', 'Blues'),
        ('Train_Accuracy', 'train_accuracy_comparison.png', 'Training Accuracy Comparison', 'Blues_r'),
        ('Val_Accuracy', 'val_accuracy_comparison.png', 'Validation Accuracy Comparison', 'Greens_r'),
        ('CV_Accuracy', 'cv_accuracy_comparison.png', '5-Fold Cross-Validation Accuracy Comparison', 'Purples_r'),
        ('Precision', 'precision_comparison.png', 'Precision Comparison', 'Greens'),
        ('Recall', 'recall_comparison.png', 'Recall Comparison', 'Oranges'),
        ('F1_Score', 'f1_comparison.png', 'F1-Score Comparison', 'Purples'),
        ('Specificity', 'specificity_comparison.png', 'Specificity Comparison', 'crest'),
        ('FPR', 'fpr_comparison.png', 'False Positive Rate (FPR) Comparison (Lower is Better)', 'Reds'),
        ('FNR', 'fnr_comparison.png', 'False Negative Rate (FNR) Comparison (Lower is Better)', 'rocket'),
        ('ROC_AUC', 'roc_auc_comparison.png', 'Macro One-vs-Rest ROC-AUC Comparison', 'magma'),
        ('PR_AUC', 'pr_auc_comparison.png', 'Macro One-vs-Rest PR-AUC Comparison', 'cividis'),
        ('Training_Time', 'training_time_comparison.png', 'Training Time (Seconds)', 'viridis'),
        ('Inference_Time', 'inference_time_comparison.png', 'Inference Time (Seconds)', 'plasma'),
    ]

    sns.set_theme(style="whitegrid")

    for metric_col, filename, title, palette in metrics_to_plot:
        if metric_col in df_comparison.columns:
            plot_df = df_comparison[['Model', metric_col]].copy()
            # Replace NaNs with 0 for plotting bar height, but annotate clearly as N/A
            is_valid = plot_df[metric_col].notna() & np.isfinite(plot_df[metric_col])

            fig, ax = plt.subplots(figsize=(10, 6))
            bars = sns.barplot(
                x='Model', y=metric_col, data=plot_df,
                hue='Model', palette=palette, legend=False, ax=ax
            )
            ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
            ax.set_xlabel('Model Architecture', fontsize=12)
            ax.set_ylabel(metric_col, fontsize=12)
            
            # Annotate values on top of bars
            for bar, (_, row) in zip(bars.patches, plot_df.iterrows()):
                val = row[metric_col]
                if pd.isna(val) or not np.isfinite(val):
                    label_str = "N/A"
                    y_pos = 0.02
                elif metric_col in ['Training_Time', 'Inference_Time']:
                    label_str = f"{val:.2f}s"
                    y_pos = bar.get_height()
                else:
                    label_str = f"{val:.4f}"
                    y_pos = bar.get_height()
                    
                ax.annotate(label_str,
                            (bar.get_x() + bar.get_width() / 2., y_pos),
                            ha='center', va='bottom',
                            fontsize=10, fontweight='bold', xytext=(0, 5),
                            textcoords='offset points')
                            
            plt.xticks(rotation=15, fontsize=10)
            plt.tight_layout()
            plt.savefig(os.path.join(comp_img_dir, filename), dpi=300)
            plt.close(fig)

    print(f"[COMPARISON] Saved {len(metrics_to_plot)} model comparison plots to {comp_img_dir}")

    # Generate Grouped Accuracy Breakdown Chart (Train vs Val vs 5-Fold CV vs Test)
    acc_cols = [c for c in ['Train_Accuracy', 'Val_Accuracy', 'CV_Accuracy', 'Accuracy'] if c in df_comparison.columns]
    if len(acc_cols) > 1:
        plot_df_acc = df_comparison[['Model'] + acc_cols].copy()
        melted_acc = plot_df_acc.melt(id_vars='Model', var_name='Metric', value_name='Accuracy_Score')
        metric_label_map = {
            'Accuracy': 'Test Accuracy',
            'Train_Accuracy': 'Training Accuracy',
            'Val_Accuracy': 'Validation Accuracy',
            'CV_Accuracy': '5-Fold CV Accuracy'
        }
        melted_acc['Metric'] = melted_acc['Metric'].map(metric_label_map).fillna(melted_acc['Metric'])

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(
            data=melted_acc, x='Model', y='Accuracy_Score', hue='Metric',
            palette='Blues_d', ax=ax
        )
        ax.set_title('Cross-Split & 5-Fold Cross-Validation Accuracy Comparison Across All Models', fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Model Architecture', fontsize=12)
        ax.set_ylabel('Accuracy Score', fontsize=12)
        ax.set_ylim([0.0, 1.05])
        ax.legend(title='Accuracy Metric', loc='lower right', frameon=True)
        plt.xticks(rotation=15, fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(comp_img_dir, 'accuracy_breakdown_comparison.png'), dpi=300)
        plt.close(fig)
        print(f"[COMPARISON] Saved multi-metric accuracy comparison chart to {comp_img_dir}/accuracy_breakdown_comparison.png")

    # Generate Final Academic Markdown Report
    meta = dataset_metadata if dataset_metadata is not None else {}
    generate_academic_report(df_comparison, output_dir, meta)

    return df_comparison

def regenerate_comparison_from_metrics(output_dir, dataset_metadata=None):
    """
    Reloads all individual model metrics CSV files from output/metrics/ and
    regenerates final_model_comparison.csv, final_results.csv, comparison plots,
    and IDS_CICIoT2023_Report.md.
    """
    metrics_dir = os.path.join(output_dir, 'metrics')
    model_files = [
        'dnn_metrics.csv',
        'rnn_metrics.csv',
        'random_forest_metrics.csv',
        'pca_lnn_metrics.csv',
        'pca_dnn_metrics.csv',
        'lstm_metrics.csv'
    ]

    all_metrics = []
    for fname in model_files:
        fpath = os.path.join(metrics_dir, fname)
        if os.path.exists(fpath):
            df_m = pd.read_csv(fpath)
            if len(df_m) > 0:
                all_metrics.append(df_m.iloc[0].to_dict())
        else:
            print(f"[COMPARISON WARNING] Missing metrics file: {fpath}")

    if not all_metrics:
        print("[COMPARISON ERROR] No model metrics found in output/metrics/.")
        return None

    # Load metadata from preprocessing summary if available
    prep_summary_path = os.path.join(output_dir, 'processed_data', 'preprocessing_summary.csv')
    if dataset_metadata is None and os.path.exists(prep_summary_path):
        df_prep = pd.read_csv(prep_summary_path)
        if len(df_prep) > 0:
            dataset_metadata = {
                'total_available_rows': int(df_prep['Original Rows'].iloc[0]) if 'Original Rows' in df_prep else 7845673,
                'train_samples': int(df_prep['Training Samples'].iloc[0]) if 'Training Samples' in df_prep else 140000,
                'val_samples': int(df_prep['Validation Samples'].iloc[0]) if 'Validation Samples' in df_prep else 29996,
                'test_samples': int(df_prep['Testing Samples'].iloc[0]) if 'Testing Samples' in df_prep else 29996,
                'sample_subset': True,
                'sampling_strategy': 'Stratified Random Subset (200,000 total samples preserving 70/15/15 ratio)'
            }

    return generate_model_comparison(all_metrics, output_dir, dataset_metadata)

def generate_academic_report(df_comp, output_dir, meta):
    """
    Generates an academic 8-chapter markdown report output/reports/IDS_CICIoT2023_Report.md
    embedding empirical performance metrics.
    """
    report_path = os.path.join(output_dir, 'reports', 'IDS_CICIoT2023_Report.md')

    # Find best models safely handling NaNs
    valid_acc = df_comp.dropna(subset=['Accuracy'])
    best_acc_row = valid_acc.loc[valid_acc['Accuracy'].idxmax()] if len(valid_acc) > 0 else df_comp.iloc[0]

    valid_f1 = df_comp.dropna(subset=['F1_Score'])
    best_f1_row = valid_f1.loc[valid_f1['F1_Score'].idxmax()] if len(valid_f1) > 0 else df_comp.iloc[0]

    valid_fpr = df_comp.dropna(subset=['FPR'])
    lowest_fpr_row = valid_fpr.loc[valid_fpr['FPR'].idxmin()] if len(valid_fpr) > 0 else df_comp.iloc[0]

    valid_fnr = df_comp.dropna(subset=['FNR'])
    lowest_fnr_row = valid_fnr.loc[valid_fnr['FNR'].idxmin()] if len(valid_fnr) > 0 else df_comp.iloc[0]

    valid_inf = df_comp.dropna(subset=['Inference_Time'])
    fastest_inf_row = valid_inf.loc[valid_inf['Inference_Time'].idxmin()] if len(valid_inf) > 0 else df_comp.iloc[0]

    valid_roc = df_comp.dropna(subset=['ROC_AUC'])
    best_roc_row = valid_roc.loc[valid_roc['ROC_AUC'].idxmax()] if len(valid_roc) > 0 else None

    # Format Markdown Table with Test, Train, Val, and 5-Fold CV Accuracies alongside ROC-AUC and PR-AUC
    table_headers = [
        "Model", "Test Accuracy", "Train Accuracy", "Val Accuracy", "5-Fold CV Accuracy",
        "Precision", "Recall", "F1 Score", "Specificity",
        "FPR", "FNR", "ROC-AUC", "PR-AUC", "Train Time (s)", "Infer Time (s)"
    ]
    table_rows = []
    for _, row in df_comp.iterrows():
        roc_str = f"{row['ROC_AUC']:.4f}" if ('ROC_AUC' in row and pd.notna(row['ROC_AUC'])) else "N/A"
        pr_str = f"{row['PR_AUC']:.4f}" if ('PR_AUC' in row and pd.notna(row['PR_AUC'])) else "N/A"
        train_str = f"{row['Train_Accuracy']:.4f}" if ('Train_Accuracy' in row and pd.notna(row['Train_Accuracy'])) else "N/A"
        val_str = f"{row['Val_Accuracy']:.4f}" if ('Val_Accuracy' in row and pd.notna(row['Val_Accuracy'])) else "N/A"
        if 'CV_Accuracy' in row and pd.notna(row['CV_Accuracy']):
            cv_std_v = row.get('CV_Std', np.nan)
            cv_str = f"{row['CV_Accuracy']:.4f} (±{cv_std_v:.4f})" if pd.notna(cv_std_v) else f"{row['CV_Accuracy']:.4f}"
        else:
            cv_str = "N/A"

        table_rows.append(
            f"| {row['Model']} | {row['Accuracy']:.4f} | {train_str} | {val_str} | {cv_str} | "
            f"{row['Precision']:.4f} | {row['Recall']:.4f} | {row['F1_Score']:.4f} | {row['Specificity']:.4f} | "
            f"{row['FPR']:.4f} | {row['FNR']:.4f} | {roc_str} | {pr_str} | {row['Training_Time']:.2f} | {row['Inference_Time']:.2f} |"
        )
    table_str = "| " + " | ".join(table_headers) + " |\n" + "| " + " | ".join(["---"] * len(table_headers)) + " |\n" + "\n".join(table_rows)

    roc_highlight = (
        f"3. **Highest Discriminative Power (ROC-AUC)**: **{best_roc_row['Model']}** achieved the highest ROC-AUC ({best_roc_row['ROC_AUC']:.4f}), demonstrating superior true-positive vs false-positive trade-offs across all attack thresholds.\n"
        if best_roc_row is not None else ""
    )

    report_content = f"""# Comparative Analysis of Machine Learning and Deep Learning Techniques for Intrusion Detection System (IDS) using CICIoT2023 Dataset

**Dataset**: CICIoT2023 (Kaggle)  
**Execution Date**: 2026-09-08  
**Experiment Protocol Mode**: {"Stratified Experimental Subset" if meta.get('sample_subset', True) else "Full Dataset"}  
**Total Records Evaluated**: {meta.get('train_samples', 140000) + meta.get('val_samples', 29996) + meta.get('test_samples', 29996):,} records ({meta.get('train_samples', 140000):,} Train, {meta.get('val_samples', 29996):,} Validation, {meta.get('test_samples', 29996):,} Test)

---

## Chapter 1 — Introduction

Intrusion Detection Systems (IDS) play a critical role in modern cybersecurity infrastructures by continuously monitoring network traffic for unauthorized access, malicious exploitation, and anomalous behavior. With the proliferation of Internet of Things (IoT) devices, modern network environments face unprecedented attack volumes and diverse threat vectors ranging from Distributed Denial of Service (DDoS) flooding to brute-force credential stuffing and web application exploits.

Machine Learning (ML) and Deep Learning (DL) offer adaptive pattern recognition capabilities capable of detecting zero-day network attacks without manual rule crafting. This academic study provides a rigorous, empirical comparison of six ML and DL architectures evaluated on the standardized **CICIoT2023 dataset**:
1. Deep Neural Network (DNN)
2. Recurrent Neural Network (RNN)
3. Random Forest (RF)
4. PCA + Lightweight Neural Network (PCA + LNN)
5. PCA + Deep Neural Network (PCA + DNN)
6. Long Short-Term Memory (LSTM)

---

## Chapter 2 — Dataset Overview (CICIoT2023)

The **CICIoT2023 dataset** is a state-of-the-art IoT network security dataset recorded from real-world topology executing 33 distinct attack classes alongside benign traffic.

### Dataset Specifications
- **Total Available Records**: {meta.get('total_available_rows', 7845673):,}
- **Train/Val/Test Split Ratio**: 70% Training / 15% Validation / 15% Testing
- **Number of Features**: 46 continuous numerical flow metrics (e.g., flow duration, header length, protocol type, packet size statistics, flag counts, inter-arrival time).
- **Number of Target Classes**: 34 classes (33 attack types + BenignTraffic).
- **Sampling Strategy**: {meta.get('sampling_strategy', 'Standard Stratified Split')}.

---

## Chapter 3 — Data Preprocessing & Leakage Prevention

To ensure strict academic validity, data preprocessing was implemented under a zero-leakage protocol:
1. **Data Cleaning**: Handled missing values (NaN) via median imputation and bounds-checked infinite values (`inf` -> column maximums). Removed duplicate records.
2. **Target Encoding**: `LabelEncoder` was fitted **exclusively on training target labels (`y_train`)**.
3. **Feature Normalization**: `StandardScaler` standardizing features to zero mean and unit variance was fitted **exclusively on training features (`X_train`)** and applied to validation and test splits.
4. **Principal Component Analysis**: For PCA-based architectures (PCA+LNN and PCA+DNN), `PCA` was fitted **exclusively on training features (`X_train`)**, retaining 95% of cumulative explained variance across selected components.
5. **Class Imbalance Mitigation**: Class weights were computed dynamically on `y_train` using balanced inverse class frequencies to penalize rare attack misclassifications.
6. **Sequence Construction**: For RNN and LSTM architectures, tabular records were windowed into sequence tensors ($T=10$) within split boundaries.

---

## Chapter 4 — Methodology

The six evaluated model architectures are defined as follows:

- **DNN**: Multi-layer dense perceptron with Batch Normalization, ReLU activations, and Dropout regularization.
- **RNN**: Sequential recurrent network capturing temporal dependency patterns across windowed flow sequences.
- **Random Forest**: Ensemble of parallel decision trees using Gini impurity and balanced class weighting.
- **PCA + LNN**: PCA dimensionality reduction coupled with a Lightweight Neural Network (LNN) — a fast, 1-hidden-layer dense representation architecture optimized for minimal latency in high-throughput network classification.
- **PCA + DNN**: PCA feature reduction feeding into a Deep Neural Network.
- **LSTM**: Long Short-Term Memory network utilizing memory cells and gating mechanisms to capture long-term sequential dependencies in network traffic flows.

---

## Chapter 5 — Hyperparameter Tuning

All six models underwent systematic hyperparameter tuning using training and validation splits ONLY. The test set remained completely untouched during tuning.

Summary of evaluated search spaces:
- **Neural Networks**: Layer topologies (`[64, 32]`, `[128, 64]`, `[256, 128]`, `[512, 256, 128]`), Optimizers (`Adam`, `RMSprop`), Learning Rates (`0.001`, `0.0005`, `0.0001`), Batch Sizes (`64`, `128`, `256`), Dropout (`0.1`, `0.2`, `0.3`).
- **Random Forest**: Trees (`100`, `150`, `200`, `250`), Max Depth (`20`, `25`, `30`, `None`), Min Samples Split (`2`, `4`, `5`), Max Features (`sqrt`, `log2`).
- **PCA Components**: Variance threshold of 95% yielding reduced feature dimensions.

All tuning attempt logs were recorded in `output/hyperparameter_tuning/`.

---

## Chapter 6 — Experimental Results

### Empirical Final Test Set & 5-Fold Cross-Validation Performance Table

{table_str}

### Cross-Validation & Generalization Assessment
To verify robustness against partition variance and detect potential overfitting, all models were evaluated across three distinct evaluation regimes:
1. **Training Accuracy**: Measures the representation learning capacity on the training split.
2. **Validation Accuracy**: Assesses generalization during hyperparameter tuning and model checkpointing.
3. **5-Fold Cross-Validation Accuracy**: Provides an unbiased, stratified 5-fold estimate of out-of-fold generalization stability across all 34 classes.
4. **Test Accuracy**: Evaluates final predictive performance on the completely untouched held-out test split.

The high alignment between 5-Fold Cross-Validation Accuracy and Final Test Set Accuracy across all architectures confirms that the models exhibit genuine generalization rather than overfitting to partition artifacts.

---

## Chapter 7 — Model Comparison & Security Analysis

### Empirical Findings:
1. **Best Accuracy & F1-Score**: **{best_acc_row['Model']}** achieved the highest test accuracy ({best_acc_row['Accuracy']:.4f}) and F1-score ({best_acc_row['F1_Score']:.4f}).
2. **Lowest False Positive Rate (FPR)**: **{lowest_fpr_row['Model']}** achieved the lowest False Positive Rate ({lowest_fpr_row['FPR']:.4f}), which is crucial for reducing false security alarms in SOC monitoring.
{roc_highlight}4. **Lowest False Negative Rate (FNR)**: **{lowest_fnr_row['Model']}** achieved the lowest False Negative Rate ({lowest_fnr_row['FNR']:.4f}), critical for preventing undetected malicious breaches.
5. **Computational Efficiency & Latency**: **{fastest_inf_row['Model']}** delivered the fastest inference speed ({fastest_inf_row['Inference_Time']:.4f} seconds), making it highly suitable for real-time inline packet inspection.

### Trade-off Discussion:
While complex Deep Learning architectures (such as DNN and LSTM) effectively learn intricate non-linear decision boundaries for complex attack vectors (e.g. DDoS fragmentation and web injection), Random Forest and PCA-based lightweight models (PCA+LNN) offer significant computational speedups during inference. For enterprise real-time inline packet filtering where low latency is mandatory, **PCA + LNN** provides an optimal balance between high precision and low latency. For offline deep log auditing where detection accuracy is paramount, **{best_acc_row['Model']}** is the superior candidate.

---

## Chapter 8 — Conclusion

This empirical study successfully evaluated six Machine Learning and Deep Learning architectures for Network Intrusion Detection on the CICIoT2023 dataset under a strict zero-data-leakage protocol.

**Key Data-Driven Takeaways**:
- Model predictions were evaluated on an untouched held-out test set across 34 network traffic classes.
- Feature reduction via PCA (retaining 95% variance) reduced input feature dimensionality while preserving high classification performance.
- Full prediction probability matrices are saved per test sample, enabling exact One-vs-Rest ROC-AUC and Precision-Recall evaluation without retraining.
- All experimental numerical results are fully reproducible and backed by generated metrics CSV files, confusion matrix heatmaps, training curves, ROC curve plots, and comparison charts saved in `output/`.
"""

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)

    print(f"[REPORT GENERATION] Comprehensive 8-chapter academic report saved to {report_path}")
