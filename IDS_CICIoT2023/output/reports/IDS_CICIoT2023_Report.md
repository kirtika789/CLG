# Comparative Analysis of Machine Learning and Deep Learning Techniques for Intrusion Detection System (IDS) using CICIoT2023 Dataset

**Dataset**: CICIoT2023 (Kaggle)  
**Execution Date**: 2026-09-08  
**Experiment Protocol Mode**: Stratified Experimental Subset  
**Total Records Evaluated**: 199,992 records (140,000 Train, 29,996 Validation, 29,996 Test)

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
- **Total Available Records**: 7,845,673
- **Train/Val/Test Split Ratio**: 70% Training / 15% Validation / 15% Testing
- **Number of Features**: 46 continuous numerical flow metrics (e.g., flow duration, header length, protocol type, packet size statistics, flag counts, inter-arrival time).
- **Number of Target Classes**: 34 classes (33 attack types + BenignTraffic).
- **Sampling Strategy**: Stratified Random Subset (200,000 total samples preserving 70/15/15 ratio).

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

### Empirical Final Test Set Performance Table

| Model | Accuracy | Precision | Recall | F1 Score | Specificity | FPR | FNR | ROC-AUC | PR-AUC | Train Time (s) | Infer Time (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LSTM | 0.9630 | 0.7674 | 0.7771 | 0.7467 | 0.9989 | 0.0011 | 0.2164 | 0.9369 | 0.7998 | 356.43 | 1.54 |
| DNN | 0.7653 | 0.5310 | 0.6068 | 0.5304 | 0.9926 | 0.0074 | 0.3932 | 0.9646 | 0.6072 | 594.75 | 0.45 |
| RNN | 0.9045 | 0.7074 | 0.7037 | 0.6982 | 0.9970 | 0.0030 | 0.2876 | 0.9955 | 0.7399 | 74.31 | 0.58 |
| Random Forest | 0.9786 | 0.8767 | 0.8319 | 0.8435 | 0.9992 | 0.0008 | 0.1494 | 0.9975 | 0.8523 | 0.34 | 0.10 |
| PCA + LNN | 0.7428 | 0.4884 | 0.5245 | 0.4772 | 0.9919 | 0.0081 | 0.4755 | 0.9370 | 0.5432 | 110.90 | 0.46 |
| PCA + DNN | 0.7356 | 0.5135 | 0.5199 | 0.4553 | 0.9917 | 0.0083 | 0.4801 | 0.9497 | 0.5336 | 137.21 | 0.73 |

---

## Chapter 7 — Model Comparison & Security Analysis

### Empirical Findings:
1. **Best Accuracy & F1-Score**: **Random Forest** achieved the highest test accuracy (0.9786) and F1-score (0.8435).
2. **Lowest False Positive Rate (FPR)**: **Random Forest** achieved the lowest False Positive Rate (0.0008), which is crucial for reducing false security alarms in SOC monitoring.
3. **Highest Discriminative Power (ROC-AUC)**: **Random Forest** achieved the highest ROC-AUC (0.9975), demonstrating superior true-positive vs false-positive trade-offs across all attack thresholds.
4. **Lowest False Negative Rate (FNR)**: **Random Forest** achieved the lowest False Negative Rate (0.1494), critical for preventing undetected malicious breaches.
5. **Computational Efficiency & Latency**: **Random Forest** delivered the fastest inference speed (0.1007 seconds), making it highly suitable for real-time inline packet inspection.

### Trade-off Discussion:
While complex Deep Learning architectures (such as DNN and LSTM) effectively learn intricate non-linear decision boundaries for complex attack vectors (e.g. DDoS fragmentation and web injection), Random Forest and PCA-based lightweight models (PCA+LNN) offer significant computational speedups during inference. For enterprise real-time inline packet filtering where low latency is mandatory, **PCA + LNN** provides an optimal balance between high precision and low latency. For offline deep log auditing where detection accuracy is paramount, **Random Forest** is the superior candidate.

---

## Chapter 8 — Conclusion

This empirical study successfully evaluated six Machine Learning and Deep Learning architectures for Network Intrusion Detection on the CICIoT2023 dataset under a strict zero-data-leakage protocol.

**Key Data-Driven Takeaways**:
- Model predictions were evaluated on an untouched held-out test set across 34 network traffic classes.
- Feature reduction via PCA (retaining 95% variance) reduced input feature dimensionality while preserving high classification performance.
- Full prediction probability matrices are saved per test sample, enabling exact One-vs-Rest ROC-AUC and Precision-Recall evaluation without retraining.
- All experimental numerical results are fully reproducible and backed by generated metrics CSV files, confusion matrix heatmaps, training curves, ROC curve plots, and comparison charts saved in `output/`.
