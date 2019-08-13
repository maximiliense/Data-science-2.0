from datascience.data.transforms.transformations import mean_patch

params = {
    'LoadCreateNN': {
        'model_params': {
            'dropout': 0.5
        }
    },
    'LoaderModule': {
        'transform': mean_patch
    }
}
