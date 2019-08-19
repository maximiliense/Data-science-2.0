# training model
from datascience.ml.metrics.metrics import ValidationAccuracyMultiple

# loading/creating model
model_params = {
    'dropout': 0.8,
    'n_labels': 34382,
    'n_input': 35
}

training_params = {
    'batch_size': 160,
    'lr': 0.1,
    'iterations': [90, 130, 150, 170, 180],
    'log_modulo': 500,
    'val_modulo': 5,
}
validation_params = {
    'metrics': (ValidationAccuracyMultiple([1, 10, 30]),)
}
# predict_params = {
#     'filters': (FilterLabelsList(get_setup_path() + '/allowed_classes_final.txt'),)
# }
