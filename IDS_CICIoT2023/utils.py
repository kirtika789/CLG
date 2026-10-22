import os
import random
import time
import numpy as np
import tensorflow as tf

def set_seeds(seed=42):
    """
    Sets random seeds across Python, NumPy, and TensorFlow for reproducibility.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print(f"[REPRODUCIBILITY] Random seed set to {seed}")

def ensure_directories(base_dir=None):
    """
    Ensures that all mandatory project and output subdirectories exist.
    """
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    
    dirs = [
        os.path.join(base_dir, "data", "raw"),
        os.path.join(base_dir, "data", "processed"),
        os.path.join(base_dir, "models"),
        os.path.join(base_dir, "tuning"),
        os.path.join(base_dir, "output", "metrics"),
        os.path.join(base_dir, "output", "hyperparameter_tuning"),
        os.path.join(base_dir, "output", "confusion_matrices"),
        os.path.join(base_dir, "output", "training_curves"),
        os.path.join(base_dir, "output", "model_comparison"),
        os.path.join(base_dir, "output", "feature_analysis"),
        os.path.join(base_dir, "output", "predictions"),
        os.path.join(base_dir, "output", "processed_data"),
        os.path.join(base_dir, "output", "reports"),
    ]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    return dirs

class Timer:
    """
    Utility timer class for measuring execution time (training and inference).
    """
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.duration = self.end - self.start
