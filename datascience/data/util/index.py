import os
import json

import re

from engine.util.console.logs import print_logs, print_errors


def get_index(index_path):
    """
    Load a label index
    :param index_path:
    :return:the index
    """
    global previous_index_path
    global previous_indexed_labels
    if 'previous_index_path' in globals() and index_path == previous_index_path:
        print_logs('Labels index in cache')
        return previous_indexed_labels

    # check if labels have been indexed
    if os.path.isfile(index_path):
        # if model is validation
        print_logs('Loading labels index ' + index_path)
        with open(index_path) as f:
            indexed_labels = json.load(f)
        indexed_labels = {int(k): int(v) for k, v in indexed_labels.items()}
        previous_index_path = index_path
        previous_indexed_labels = indexed_labels

    else:
        print_errors('index ' + index_path + ' does not exist...')
        indexed_labels = None
    return indexed_labels


def reload_index(index_path):
    """
    cancel previous loading and load the new index
    :param index_path:
    :return:
    """
    global previous_index_path
    previous_index_path = None
    return get_index(index_path)


def reverse_indexing(index, column=0):
    """
    Switch keys and values of the index.. If list of index, change reverse the column chosen
    :param index:
    :param column:
    :return:
    """
    reversed_index = {}
    d = index if type(index) is dict else index[column]
    for k, v in d.items():
        reversed_index[v] = k

    return reversed_index


def save_reversed_index(path, index, column=0):
    """
    save the index on disk for future use
    :param path:
    :param index:
    :param column:
    :return:
    """
    print_logs('Saving index at ' + path)
    reversed_index = reverse_indexing(index, column)
    _json = json.dumps(reversed_index)
    f = open(path, "w")
    f.write(_json)
    f.close()


def get_to_save(save_index):
    if type(save_index) is str and ('save' in save_index or 'default' == save_index):
        save_index = True
    elif type(save_index) is not bool:
        save_index = False
    return save_index


def get_to_load(save_index):
    return type(save_index) is str and 'load' in save_index

