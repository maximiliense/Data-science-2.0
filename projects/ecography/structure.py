from datascience.data.transforms.transformations import normalize

params = {
    'LoadCreateNN': {
        'model_params': {
            'dropout': 0.8
        }
    },
    'LoaderModule': {
        'transform': normalize
    }
}
