"""
Hyperparameter Search Space Definitions for IDS CICIoT2023 Models.
"""

DNN_GRID = [
    {'layers': [256, 128], 'lr': 0.001, 'batch_size': 128, 'opt': 'adam', 'dropout': 0.2},
    {'layers': [512, 256, 128], 'lr': 0.001, 'batch_size': 128, 'opt': 'adam', 'dropout': 0.3},
    {'layers': [128, 64], 'lr': 0.0005, 'batch_size': 64, 'opt': 'rmsprop', 'dropout': 0.2},
    {'layers': [256, 128, 64], 'lr': 0.0001, 'batch_size': 128, 'opt': 'adam', 'dropout': 0.2},
]

RNN_GRID = [
    {'units': [64, 32], 'lr': 0.001, 'batch_size': 128, 'opt': 'adam', 'dropout': 0.2},
    {'units': [128, 64], 'lr': 0.001, 'batch_size': 128, 'opt': 'adam', 'dropout': 0.3},
    {'units': [64], 'lr': 0.0005, 'batch_size': 64, 'opt': 'rmsprop', 'dropout': 0.2},
    {'units': [128, 64, 32], 'lr': 0.0001, 'batch_size': 128, 'opt': 'adam', 'dropout': 0.2},
]

RF_GRID = [
    {'n_estimators': 100, 'max_depth': 20, 'min_samples_split': 2, 'max_features': 'sqrt'},
    {'n_estimators': 200, 'max_depth': 30, 'min_samples_split': 5, 'max_features': 'sqrt'},
    {'n_estimators': 150, 'max_depth': None, 'min_samples_split': 2, 'max_features': 'log2'},
    {'n_estimators': 250, 'max_depth': 25, 'min_samples_split': 4, 'max_features': 'sqrt'},
]

PCA_LNN_GRID = [
    {'units': 128, 'lr': 0.001, 'batch_size': 128, 'opt': 'adam', 'dropout': 0.1, 'pca_variance': 0.95},
    {'units': 256, 'lr': 0.001, 'batch_size': 128, 'opt': 'adam', 'dropout': 0.2, 'pca_variance': 0.95},
    {'units': 64, 'lr': 0.0005, 'batch_size': 64, 'opt': 'rmsprop', 'dropout': 0.1, 'pca_variance': 0.90},
    {'units': 128, 'lr': 0.0001, 'batch_size': 256, 'opt': 'adam', 'dropout': 0.1, 'pca_variance': 0.99},
]

PCA_DNN_GRID = [
    {'layers': [256, 128], 'lr': 0.001, 'batch_size': 128, 'opt': 'adam', 'dropout': 0.2, 'pca_variance': 0.95},
    {'layers': [512, 256, 128], 'lr': 0.001, 'batch_size': 128, 'opt': 'adam', 'dropout': 0.3, 'pca_variance': 0.95},
    {'layers': [128, 64], 'lr': 0.0005, 'batch_size': 64, 'opt': 'rmsprop', 'dropout': 0.2, 'pca_variance': 0.90},
    {'layers': [256, 128, 64], 'lr': 0.0001, 'batch_size': 128, 'opt': 'adam', 'dropout': 0.2, 'pca_variance': 0.99},
]

LSTM_GRID = [
    {'units': [64, 32], 'lr': 0.001, 'batch_size': 128, 'opt': 'adam', 'dropout': 0.2},
    {'units': [128, 64], 'lr': 0.001, 'batch_size': 128, 'opt': 'adam', 'dropout': 0.3},
    {'units': [64], 'lr': 0.0005, 'batch_size': 64, 'opt': 'rmsprop', 'dropout': 0.2},
    {'units': [128, 64, 32], 'lr': 0.0001, 'batch_size': 128, 'opt': 'adam', 'dropout': 0.2},
]
