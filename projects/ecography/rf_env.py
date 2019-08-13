from engine.dataset.datasets.dataset_glc import GeoLifeClefDataset
from engine.modules import LoaderModule
from engine.modules.load_create_skl import LoadCreateRF
from engine.modules.run_skl_train import RunTraining
from engine.training_validation.metrics import ValidationAccuracyMultipleBySpecies, ValidationMRRBySpecies, ValidationAccuracyRangeBySpecies

wf = [
    (
        LoadCreateRF(), {
            'n_estimators': 50,
            'max_depth': 10
        }
    ),
    (
        LoaderModule(),
        {
            'dataset_class': GeoLifeClefDataset,
            '_source': 'gbif',
            'validation_size': 0,
            'test_size': 0.1,
        }
    ),
    (
        RunTraining(),
        {
            # 'param_grid': {
            #     'max_depth': [3],
            #     'max_features': [2],
            #     'min_samples_split': [3],
            #     'bootstrap': [True],
            #     'criterion': ['entropy']},
            'training_params': {
                'metrics': (ValidationAccuracyMultipleBySpecies([1, 10, 30]), ValidationMRRBySpecies(), ValidationAccuracyRangeBySpecies())
            }
        }
    )
]
