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

### 1. Requirements & Environment
Activate the pre-configured virtual environment:

```powershell
# Windows PowerShell:
.\ids_env\Scripts\Activate.ps1
```

Or install dependencies manually via `pip`:

```bash
pip install -r requirements.txt
```

### 2. Run Verification Suite (Optional)
To verify all multiclass ROC-AUC calculations, confusion matrix handlers, and models on synthetic data:

```bash
python verify_roc_auc.py
```

### 3. Execution Options

#### A. Generate Comparison Charts & Academic Report (Fastest — 2 seconds)
Regenerates all 11 high-resolution comparison plots and the comprehensive 8-chapter academic markdown report without retraining:

```bash
python main.py --report-only
```

#### B. Fast Test Run with a Single Model (e.g., Random Forest or DNN)
Run hyperparameter tuning, training, and evaluation on a custom subset size:

```bash
# Fast test run with Random Forest (5,000 samples ~ 5 seconds)
python main.py --model rf --sample-size 5000

# Run specific architecture (options: dnn, rnn, rf, pca_lnn, pca_dnn, lstm)
python main.py --model dnn --sample-size 50000
```

#### C. Full Pipeline End-to-End (All 6 Models)
To run the full end-to-end benchmark with all 6 models:

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
