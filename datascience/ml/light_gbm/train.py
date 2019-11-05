import lightgbm as lgb

from datascience.ml.evaluation import validate, export_results
from engine.core import module
from engine.logging import print_errors, print_h1, print_notification
from engine.path import output_path
from engine.util.log_email import send_email
from engine.util.log_file import save_file
from engine.util.merge_dict import merge_smooth
from engine.parameters import special_parameters


@module
def fit(train, test, validation=None, validation_params=None, export_params=None, model_name='model', **kwargs):
    """
    Fit a light GBM model. If validation_only or export is True, then the training is not performed and the model is
    loaded.
    :param model_name:
    :param export_params:
    :param validation_params:
    :param train:
    :param test:
    :param validation:
    :param kwargs:
    :return:
    """

    nb_labels = _nb_labels(train, test, validation)

    train_data = _to_lgb_dataset(train)
    test_data = _to_lgb_dataset(test)
    val_data = test_data if validation is None else _to_lgb_dataset(validation)

    if not (special_parameters.validation_only or special_parameters.export):
        print_h1('Training: ' + special_parameters.setup_name)
        num_round = 10

        param = kwargs
        merge_smooth(param, _default_params)
        param['num_class'] = nb_labels

        bst = lgb.train(param, train_data, num_round, valid_sets=[val_data])
        bst.save_model(output_path('models/{}.bst'.format(model_name)))
    else:
        bst = lgb.Booster(model_file=output_path('models/{}.bst'.format(model_name)))

    print_h1('Validation/Export: ' + special_parameters.setup_name)

    testset, labels = test.numpy()
    predictions = bst.predict(testset)

    # validation
    if special_parameters.validation_only or not special_parameters.export:
        res = validate(predictions, labels, **({} if validation_params is None else validation_params), final=True)

        print_notification(res, end='')

        if special_parameters.mail >= 1:
            send_email('Final results for XP ' + special_parameters.setup_name, res)
        if special_parameters.file:
            save_file(output_path('validation.txt'), 'Final results for XP ' + special_parameters.setup_name, res)

    if special_parameters.export:
        export_results(test, predictions, **({} if export_params is None else export_params))


def _to_lgb_dataset(dataset):
    if not hasattr(dataset, 'numpy'):
        print_errors(str(type(dataset)) + ' must implement the numpy method...', do_exit=True)
    data, label = dataset.numpy()
    return lgb.Dataset(data, label=label)


def _nb_labels(train, test, val):
    max_labels = max(max(train.labels), max(test.labels))
    if val is not None:
        max_labels = max(max_labels, max(val.labels))
    return max_labels + 1


_default_params = {
    'num_leaves': 31,
    'objective': 'multiclass'
}
