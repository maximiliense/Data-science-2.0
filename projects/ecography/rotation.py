from datascience.data.transforms.transformations import random_rotation

params = {
    'LoadCreateNN': {
        'model_params': {
            'dropout': 0.7
        }
    },
    'LoaderModule': {
        'transform': random_rotation
    }
}
