from datascience.data.transforms.transformations import permutation

config = {
    'load_create_nn': {
        'model_params': {
            'dropout': 0.7
        }
    },
    'occurrence_loader': {
        'transform': permutation
    }
}
