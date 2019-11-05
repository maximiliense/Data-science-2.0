from datascience.ml.metrics import ValidationAccuracyMultipleBySpecies, ValidationMRRBySpecies, JustExportPredictions
from datascience.ml.metrics import ValidationAccuracyRangeBySpecies, ValidationAccuracyForAllSpecies
from datascience.ml.neural.models import load_create_nn
from datascience.ml.neural.models.inception_env_coocs import InceptionEnvCoocs
from datascience.data.loader import occurrence_loader
from datascience.data.datasets.dataset_rasters_and_cooccurrences_rtree import GeoLifeClefDataset
from datascience.ml.neural.supervised import fit
from datascience.ml.neural.supervised.predict.predict_grid import predict_grid
from datascience.ml.evaluation.export import export_results
from datascience.ml.evaluation.filters import FilterLabelsList
from datascience.tools.knn_tools.knn_index import extract_cooccurrences_multipoints

model_params = {
                'dropout': 0.8,
                'n_labels': 3336,
                'n_input': 77,
                'config': 0
            }

# loading/creating model
model = load_create_nn(model_class=InceptionEnvCoocs, model_params=model_params)

# loading dataset
_, _, test = occurrence_loader(GeoLifeClefDataset, source='glc19_test',
                               test_size=1, id_name='glc19TestOccId', label_name=None)

train, _, _ = occurrence_loader(GeoLifeClefDataset, source='glc18', test_size=0.0, validation_size=0.0,
                                id_name='patch_id', label_name='species_glc_id', second_neihbour=False)

extract_cooccurrences_multipoints(train, test, leaf_size=2, validation=None, nb_labels=3336)

predictions = predict_grid(model, test, batch_size=128, features_activation=False, logit=False)

f = FilterLabelsList('/home/benjamin/pycharm/Data-science-2.0/projects/best_of_labs/allowed_classes.txt')

f(predictions)

export_results(test, predictions)