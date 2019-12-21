import numpy as np


def validate(predictions, labels, metrics=tuple(), final=False, statistics=None, validation_id=None):
    """
    :param final:
    :param predictions:
    :param labels:
    :param statistics:
    :param validation_id:
    :param metrics: a tuple of metrics
    :return:
    """
    if len(metrics) == 0:
        return ''

    result = '\n```\n'
    result += '          -------------------\n'
    for metric in metrics:
        if not metric.sort_needed and (final or not metric.final_validation):
            score, desc = metric(predictions, labels)
            result += desc + '\n'
            result += '          -------------------\n'

    predictions = np.argsort(-predictions, axis=1)

    for metric in metrics:
        if metric.sort_needed and (final or not metric.final_validation):
            score, desc = metric(predictions, labels)
            result += desc + '\n'
            result += '          -------------------\n'

    result += '```\n\n'

    if statistics is not None:
        statistics.update_statistics(metrics, validation_id)

    return result
