import lightgbm as lgb

from engine.logging import print_errors


def fit(dataset):
    if not hasattr(dataset, 'numpy'):
        print_errors(str(type(dataset)) + ' must implement the numpy method...', do_exit=True)

    data, label = dataset.numpy()
    train_data = lgb.Dataset(data, label=label)
