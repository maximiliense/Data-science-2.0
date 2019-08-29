from datascience.data.transforms.transformations import normalize

params = {
    'load_create_nn': {
        'model_params': {
            'dropout': 0.8
        }
    },
    'occurrence_loader': {
        'transform': normalize
    }
}
