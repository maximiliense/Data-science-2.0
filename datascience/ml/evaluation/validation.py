import numpy as np


def validate(predictions, labels, metrics=tuple(), final=False):
    """
    :param final:
    :param predictions:
    :param labels:
    :param metrics: a tuple of metrics
    :return:
    """
    if len(metrics) == 0:
        return ''

    result = '\n```\n'
    result += '          -------------------\n'
    for metric in metrics:
        if not metric.sort_needed and (final or not metric.final_validation):
            result += str(metric(predictions, labels)) + '\n'
            result += '          -------------------\n'

    predictions = np.argsort(-predictions, axis=1)

    for metric in metrics:
        if metric.sort_needed and (final or not metric.final_validation):
            result += str(metric(predictions, labels)) + '\n'
            result += '          -------------------\n'

    result += '```\n\n'
    return result
