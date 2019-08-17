model_params = {
    'dropout': 0.7,
    'n_labels': 4520,
    'n_input': 77
}

training_params = {
    'batch_size': 128,
    'lr': 0.1,
    'iterations': [90, 130, 150, 170, 180],
    'log_modulo': 200,
    'val_modulo': 5,
}
