from datascience.ml.metrics import ValidationAccuracyMultipleBySpecies, ValidationMRRBySpecies, JustExportPredictions
from datascience.ml.metrics import ValidationAccuracyRangeBySpecies, ValidationAccuracyForAllSpecies
from datascience.ml.neural.models import load_create_nn, InceptionEnv
from datascience.data.loader import occurrence_loader
from datascience.data.datasets import EnvironmentalDataset
from datascience.ml.neural.supervised import fit
from datascience.ml.neural.supervised.predict.predict_grid import predict_grid
from datascience.ml.evaluation.export import export_results
from datascience.ml.evaluation.filters import FilterLabelsList

from projects.best_of_labs.configs.inception import model_params, training_params

# loading/creating model
model = load_create_nn(model_class=InceptionEnv, model_params=model_params)

# loading dataset
_, _, test = occurrence_loader(EnvironmentalDataset, source='glc19_test',
                               test_size=1, id_name='glc19TestOccId', label_name=None)

predictions = predict_grid(model, test, batch_size=128, features_activation=False, logit=False)

f = FilterLabelsList('/home/benjamin/pycharm/Data-science-2.0/projects/best_of_labs/allowed_classes.txt')

f(predictions)

export_results(test, predictions)
