from engine.parameters import special_parameters
import os
import json

from engine.logging import print_errors


def check_source(source_name):
    root_path = os.path.join(special_parameters.root_path, special_parameters.source_path, source_name + '.json')
    if not os.path.isfile(root_path):
        print_errors('the source ' + source_name + ' does not exist...', do_exit=True)

    with open(root_path) as f:
        d = json.load(f)

    if special_parameters.machine not in d:
        print_errors('The source ' + source_name + ' is not available for ' + str(special_parameters.machine),
                     do_exit=True)

    results = {}

    for k, v in d[special_parameters.machine].items():
        if not k.startswith('_'):
            results[k] = v
    results['source_name'] = source_name

    return results
