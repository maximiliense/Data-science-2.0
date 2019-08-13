from datascience.ml.metrics import ValidationAccuracyMultipleBySpecies, ValidationMRRBySpecies
from datascience.ml.metrics import ValidationAccuracyRangeBySpecies, ValidationAccuracyForAllSpecies
from datascience.ml.neural.models import load_create_nn, InceptionEnv
from datascience.data.loader import occurrence_loader
from datascience.data.datasets import EnvironmentalDataset
from datascience.ml.neural.supervised import fit
from datascience.data.model_selection import train_test_split_stratified

# loading/creating model
model_params = {
    'dropout': 0.7,
    'n_labels': 4520,
    'n_input': 77
}
model = load_create_nn(model_class=InceptionEnv, model_params=model_params)

# loading dataset
train, val, test = occurrence_loader(
    EnvironmentalDataset,
    source='gbif_taxref',
    splitter=train_test_split_stratified,
    id_name='id',
    label_name='Label'
)

# training model
training_params = {
    'batch_size': 128,
    'lr': 0.1,
    'iterations': [90, 130, 150, 170, 180],
    'log_modulo': 200,
    'val_modulo': 5,
}
validation_params = {
    'metrics': (ValidationAccuracyMultipleBySpecies([1, 10, 30]), ValidationMRRBySpecies(),
                ValidationAccuracyRangeBySpecies(max_top_k=100, final_validation=True),
                ValidationAccuracyForAllSpecies(train=train, final_validation=True))
}
# predict_params = {
#     'filters': (FilterLabelsList(get_setup_path() + '/allowed_classes.txt'),)
# }
fit(
    model, train=train, val=val, test=test, training_params=training_params, validation_params=validation_params
)
