from datascience.data.transforms.transformations import constant_patch

params = {
    'load_create_nn': {
        'model_params': {
            'dropout': 0.5
        }
    },
    'occurrence_loader': {
        'transform': constant_patch
    }
}
