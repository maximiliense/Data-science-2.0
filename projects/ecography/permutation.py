from datascience.data.transforms.transformations import permutation

params = {
    'LoadCreateNN': {
        'model_params': {
            'dropout': 0.7
        }
    },
    'LoaderModule': {
        'transform': permutation
    }
}
