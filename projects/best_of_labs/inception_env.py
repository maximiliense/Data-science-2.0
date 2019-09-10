from datascience.ml.metrics import ValidationAccuracyMultipleBySpecies, ValidationMRRBySpecies
from datascience.ml.metrics import ValidationAccuracyRangeBySpecies, ValidationAccuracyForAllSpecies
from datascience.ml.neural.models import load_create_nn, InceptionEnv
from datascience.data.loader import occurrence_loader
from datascience.data.datasets import EnvironmentalDataset
from datascience.ml.neural.supervised import fit
from datascience.ml.neural.supervised.predict import predict
from datascience.model_selection import train_test_split_stratified
from datascience.ml.neural.loss.loss import CELoss
from datascience.ml.evaluation.export import export_results

from projects.ecography.configs.inception import model_params, training_params

# loading/creating model
model = load_create_nn(model_class=InceptionEnv, model_params=model_params)

# loading dataset
_, _, test = occurrence_loader(EnvironmentalDataset, source='glc19_test', splitter=train_test_split_stratified, test_size=1)

print(type(test))
predictions, labels, _ = predict(model, test, CELoss, export=True)

export_results(test, predictions)