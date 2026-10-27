import os
import sys
import argparse
import pandas as pd

# Ensure project root is in sys.path for direct execution
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from tuning.hyperparams import (
        DNN_GRID, RNN_GRID, RF_GRID,
        PCA_LNN_GRID, PCA_DNN_GRID, LSTM_GRID
    )
except ImportError:
    from hyperparams import (
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
        key = model_name.lower().replace(' ', '_').replace('+', '')
        if key == 'rf':
            key = 'random_forest'
        filename = f"{key}_tuning_results.csv"
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
            'rf': RF_GRID,
            'random_forest': RF_GRID,
            'pca_lnn': PCA_LNN_GRID,
            'pca_dnn': PCA_DNN_GRID,
            'lstm': LSTM_GRID
        }
        key = model_name.lower().replace('+', '').replace(' ', '_')
        while '__' in key:
            key = key.replace('__', '_')
        key = key.strip('_')
        return mapping.get(key, [])

def main():
    parser = argparse.ArgumentParser(description="Hyperparameter Tuning Inspection & Execution Tool")
    parser.add_argument('--model', type=str, default='all', choices=['all', 'rf', 'random_forest', 'dnn', 'rnn', 'lstm', 'pca_lnn', 'pca_dnn'], help="Model to inspect or tune")
    parser.add_argument('--run', action='store_true', help="Execute tuning experiment for the selected model via main.py")
    parser.add_argument('--sample-size', type=int, default=5000, help="Sample size if --run is specified (default: 5000)")
    args = parser.parse_args()

    project_root = parent_dir
    output_dir = os.path.join(project_root, "output")
    tuner = HyperparameterTuner(output_dir)

    print("=" * 80)
    print("      IDS CICIoT2023 — HYPERPARAMETER TUNING SEARCH SPACE MANAGER       ")
    print("=" * 80)
    print(f"Output Directory: {tuner.tuning_dir}\n")

    models_to_display = (
        ['Random Forest', 'DNN', 'RNN', 'LSTM', 'PCA + LNN', 'PCA + DNN']
        if args.model == 'all'
        else [args.model]
    )

    for m in models_to_display:
        grid = tuner.get_search_space(m)
        print(f"[{m.upper()}] Search Space ({len(grid)} Configurations):")
        for idx, cfg in enumerate(grid, 1):
            print(f"  Config {idx}: {cfg}")
        print("-" * 80)

    if args.run:
        target_model = 'rf' if args.model in ['all', 'random_forest', 'rf'] else args.model
        print(f"\n[EXECUTION] Launching experiment for model: {target_model} (sample_size={args.sample_size:,})...")
        from main import main as run_pipeline
        sys.argv = ['main.py', '--model', target_model, '--sample-size', str(args.sample_size)]
        run_pipeline()
    else:
        print("\nTip: To run the tuning experiment for Random Forest, run:")
        print("  python tuning/tuner.py --model rf --run --sample-size 5000")
        print("or:")
        print("  python main.py --model rf --sample-size 5000\n")

if __name__ == '__main__':
    main()
