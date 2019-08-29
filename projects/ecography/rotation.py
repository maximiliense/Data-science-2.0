from datascience.data.transforms.transformations import random_rotation

params = {
    'load_create_nn': {
        'model_params': {
            'dropout': 0.7
        }
    },
    'occurrence_loader': {
        'transform': random_rotation
    }
}
