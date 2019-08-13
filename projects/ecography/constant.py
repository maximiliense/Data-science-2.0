from datascience.data.transforms.transformations import constant_patch

params = {
    'LoadCreateNN': {
        'model_params': {
            'dropout': 0.5
        }
    },
    'LoaderModule': {
        'transform': constant_patch
    }
}
