from datascience.ml.metrics.metrics import ValidationAccuracyMultipleBySpecies, ValidationMRRBySpecies
from datascience.ml.metrics.metrics import ValidationAccuracyRangeBySpecies, ValidationAccuracyForAllSpecies
from datascience.data.loader import occurrence_loader
from datascience.data.datasets import EnvironmentalDataset
from datascience.ml.xgboost.train import fit
from engine.parameters.special_parameters import get_parameters

max_depth = get_parameters('max_depth', 2)

# loading dataset
train, _, test = occurrence_loader(
    EnvironmentalDataset, source='gbif_taxref', validation_size=0, size_patch=1
)

# training model
training_params = {
    'metrics': (ValidationAccuracyMultipleBySpecies([1, 10, 30]), ValidationMRRBySpecies(),
                ValidationAccuracyRangeBySpecies(max_top_k=100, final_validation=True),
                ValidationAccuracyForAllSpecies(train=train, final_validation=True))
}
fit(train=train, test=test, training_params=training_params,
    objective='multi:softprob', max_depth=max_depth, seed=4242, eval_metric='merror', num_class=4520,
    num_boost_round=360, early_stopping_rounds=10, verbose_eval=1, updater='grow_gpu',
    predictor='gpu_predictor', tree_method='gpu_hist')
