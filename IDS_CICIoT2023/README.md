# Intrusion Detection System (IDS) — CICIoT2023 Benchmark Project

A complete, rigorous, and reproducible academic Machine Learning (ML) and Deep Learning (DL) project evaluating six model architectures on the **CICIoT2023 dataset from Kaggle**.

---

## 📁 Project Structure

```text
IDS_CICIoT2023/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── dnn.py
│   ├── rnn.py
│   ├── random_forest.py
│   ├── pca_lnn.py
│   ├── pca_dnn.py
│   └── lstm.py
│
├── tuning/
│
├── output/
│   ├── metrics/
│   ├── hyperparameter_tuning/
│   ├── confusion_matrices/
│   ├── training_curves/
│   ├── model_comparison/
│   ├── feature_analysis/
│   ├── predictions/
│   ├── processed_data/
│   └── reports/
│
├── preprocessing.py
├── evaluation.py
├── comparison.py
├── utils.py
├── main.py
├── requirements.txt
└── README.md
```

---

## 🚀 Installation & Execution

### 1. Requirements
Install dependencies using `pip`:

```bash
pip install -r requirements.txt
```

### 2. Run Pipeline End-to-End
To run the complete pipeline with hyperparameter tuning, model training, test evaluation, comparison charts, and report generation:

```bash
# Default mode: Stratified Experimental Subset (200,000 samples)
python main.py

# Full dataset mode (7.85 million samples):
python main.py --full-dataset
```

---

## 📊 Models Evaluated
1. **DNN**: Deep Neural Network
2. **RNN**: Recurrent Neural Network
3. **Random Forest**: Random Forest Ensemble
4. **PCA + LNN**: Principal Component Analysis + Lightweight Neural Network
5. **PCA + DNN**: Principal Component Analysis + Deep Neural Network
6. **LSTM**: Long Short-Term Memory Network

---

## 🔒 Data Leakage Prevention Protocol
All preprocessing transformers (`StandardScaler`, `PCA`, `LabelEncoder`, class weights) are fitted **EXCLUSIVELY on training data (`X_train`)**. The validation and held-out test set (`X_test`) remain completely untouched until final evaluation.

---

## 📑 Generated Outputs (`output/`)
- `output/metrics/`: Individual model metrics CSVs and `final_model_comparison.csv`
- `output/hyperparameter_tuning/`: Tuning results across all search spaces
- `output/confusion_matrices/`: PNG heatmaps and numerical CSV confusion matrices
- `output/training_curves/`: Training vs Validation accuracy and loss curves
- `output/model_comparison/`: 9 comparative bar charts (Accuracy, Precision, Recall, F1, Specificity, FPR, FNR, Training Time, Inference Time)
- `output/feature_analysis/`: Random Forest feature importance and PCA explained variance
- `output/predictions/`: Model predictions CSVs with probabilities
- `output/processed_data/`: Preprocessing summary CSV
- `output/reports/`: Complete 8-chapter academic markdown report (`IDS_CICIoT2023_Report.md`)
