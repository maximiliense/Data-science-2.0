from engine.logging.logs import print_statistics

import torch

import progressbar
import warnings

import pandas as pd

from datascience.data.util.index import get_index
from engine.path import output_path
from engine.parameters import special_parameters
from engine.core import module

import numpy as np

from engine.logging.logs import print_logs


def export_results(dataset, predictions, size=50, header=False):
    order = np.argsort(-predictions, axis=1)
    results = []

    export_path = output_path('predictions.csv')

    # check if labels have been indexed
    index_path = output_path('index.json')

    indexed_labels = get_index(index_path)

    for i in range(order.shape[0]):
        for j in range(size):
            jth = order[i][j]
            proba = predictions[i][jth]
            if indexed_labels is None:
                class_id = jth
            else:
                if jth in indexed_labels:
                    class_id = indexed_labels[jth]
                else:
                    continue
            _id = int(dataset.ids[i])
            results.append([_id, class_id, j+1, proba])

    df = pd.DataFrame(data=results, columns=['id', 'class_id', 'rank', 'proba'])
    df.to_csv(export_path, sep=';', header=header, index=False)
    print_statistics('Predictions saved at: ' + export_path)


exported_count = 0


def _export_bigdata(f, results, test, indexed_labels, size):
    """
    Exporting a batch of results
    :param offset:
    :param size:
    :param f:
    :param results:
    :param test:
    :param indexed_labels:
    :return:
    """

    global exported_count
    results = np.concatenate(results)
    order = np.argsort(-results, axis=1)[:, :size]
    output = []
    c = 0
    for i, elmt in enumerate(order):
        _id = test.ids[i + exported_count]  # _id = int(test.ids[i])
        c += 1
        for j in elmt:
            proba = results[i][j]
            if indexed_labels is None:
                class_id = j
            else:
                if j in indexed_labels:
                    class_id = indexed_labels[j]
                else:
                    continue
            output.append([_id, class_id, j+1, proba])
    df = pd.DataFrame(np.array(output))
    df.to_csv(f, header=False, index=False)
    exported_count += c


@module
def export_bigdata(model, test, batch_size, buffer_size, size):
    num_workers = special_parameters.nb_workers
    test_loader = torch.utils.data.DataLoader(test, shuffle=False, batch_size=batch_size, num_workers=num_workers)

    results = []

    model.eval()
    export_path = output_path('predictions.csv')
    # check if labels have been indexed
    index_path = output_path('index.json')

    indexed_labels = get_index(index_path)

    with open(export_path, 'w') as f:
        print_logs('Exporting predictions at ' + export_path)
        f.write('id,class_id,rank,proba\n')  # header

        warnings.simplefilter('ignore')  # warning because old import in progressbar
        bar = progressbar.ProgressBar(max_value=len(test_loader))
        warnings.simplefilter('default')
        for idx, data in enumerate(test_loader):
            # get the inputs
            inputs, labels = data

            outputs = model(inputs)

            results.append(outputs.detach().cpu().numpy())
            if len(results) >= buffer_size:
                _export_bigdata(f, results, test, indexed_labels, size)
                results = []
            bar.update(idx)
        if len(results) >= 0:
            _export_bigdata(f, results, test, indexed_labels, size)
        bar.finish()
