import os
import pandas as pd
from tuning.hyperparams import (
    DNN_GRID, RNN_GRID, RF_GRID,
    PCA_LNN_GRID, PCA_DNN_GRID, LSTM_GRID
)

class HyperparameterTuner:
    """
    Hyperparameter Tuning Manager for IDS CICIoT2023 benchmark experiments.
    Ensures all tuning iterations are performed exclusively on training and validation splits.
    """
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.tuning_dir = os.path.join(output_dir, "hyperparameter_tuning")
        os.makedirs(self.tuning_dir, exist_ok=True)

    def save_tuning_results(self, model_name, tuning_results_list):
        """
        Saves tuning results list to output/hyperparameter_tuning/{model_name}_tuning_results.csv.
        """
        filename = f"{model_name.lower().replace(' ', '_').replace('+', '')}_tuning_results.csv"
        filepath = os.path.join(self.tuning_dir, filename)
        df = pd.DataFrame(tuning_results_list)
        df.to_csv(filepath, index=False)
        print(f"[TUNING MANAGER] Saved {model_name} search results to {filepath}")
        return filepath

    def get_search_space(self, model_name):
        """
        Returns search space configuration grid for a given model.
        """
        mapping = {
            'dnn': DNN_GRID,
            'rnn': RNN_GRID,
            'random_forest': RF_GRID,
            'pca_lnn': PCA_LNN_GRID,
            'pca_dnn': PCA_DNN_GRID,
            'lstm': LSTM_GRID
        }
        key = model_name.lower().replace(' ', '_').replace('+', '')
        return mapping.get(key, [])
