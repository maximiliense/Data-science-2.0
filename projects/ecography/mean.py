from datascience.data.transforms.transformations import mean_patch

config = {
    'load_create_nn': {
        'model_params': {
            'dropout': 0.5
        }
    },
    'occurrence_loader': {
        'transform': mean_patch
    }
}
