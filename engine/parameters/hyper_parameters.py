import json
import os
import sys
import traceback

from engine.parameters import special_parameters
from engine.util.console.logs import print_errors

parameters = {
    'dropout': 'load_create_nn.model_params.dropout',
    'batch_size': 'fit.training_params.batch_size',
    'lr': 'fit.training_params.lr',
    'architecture': 'load_create_nn.model_params.architecture',
    'last_sigmoid': 'load_create_nn.model_params.last_sigmoid',
    'source': 'occurrence_loader.source',
    'limit': 'occurrence_loader.limit',
    'latitude': 'pplot.latitude',
    'longitude': 'pplot.longitude'
}

# this object is use to override modules keywords parameters
_overriding_parameters = {}


def overriding_parameters():
    global _overriding_parameters
    return _overriding_parameters


def check_aliases(args):
    params = args.params
    if args.validation_only:
        special_parameters.validation_only = True
    if args.export:
        special_parameters.export = True
    return params


def _recursive_setup_config(keys, config, value, rec=0):
    if len(keys) == rec + 1:
        config[keys[rec]] = value
    else:
        if keys[rec] not in config:
            config[keys[rec]] = {}
        _recursive_setup_config(keys, config[keys[rec]], value, rec+1)


def check_parameters(args):
    params = check_aliases(args)
    try:
        if special_parameters.check_known_parameters:
            for k, v in parameters.items():
                if '.' + k not in params and k in params:
                    params = params.replace(k, v)

        params = params if len(params) > 0 else ''

        last_error = None
        error = False
        initial_error = None
        while (last_error is None or error) and type(params) is str:
            try:
                params = eval("{" + params + "}")

            # tries to correct the error automatically if an exception is raised
            except NameError as e:
                if last_error is None:
                    initial_error = str(e)
                if str(e) == last_error:
                    print_errors(initial_error, do_exit=True)
                last_error = str(e)
                error = True

                import re
                k = e.args[0].replace('name \'', '').replace('\' is not defined', '')
                # extract the full key from the params line
                regex = '(?<![\'"])' + k + '[^:]*(?=:)'
                patterns = re.findall(regex, params)

                for p in patterns:
                    params = params.replace(p, '"' + p + '"')
        new_params = {}
        for k, v in params.items():
            k_split = k.split('.')
            c_param = new_params
            for i, k2 in enumerate(k_split):
                if i < len(k_split) - 1:
                    if k2 not in c_param:
                        c_param[k2] = {}
                    c_param = c_param[k2]

            c_param[k2] = v
        # when additional config used..... Not yet
        for k, v in params.items():
            key_split = k.split('.')
            _recursive_setup_config(key_split, {}, v)
    except json.decoder.JSONDecodeError as e:
        traceback.print_exc()
        print_errors(e)
        print_errors('Error in the parameters: ' + params, do_exit=True)
    global _overriding_parameters
    _overriding_parameters = new_params


def _check_config(config, params):
    for k, v in config.items():
        if k in params and type(v) is dict and type(params[k]) is dict:
            _check_config(config[k], params[k])
        else:
            params[k] = v


def check_config(args):
    global _overriding_parameters
    if args.config is not None:
        config = None
        for config in args.config:
            path = os.path.join(os.path.dirname(sys.argv[0]), config)
            if os.path.isfile(path + '.py'):
                imported_file = __import__(path.replace('/', '.'), globals(), locals(), ['config'])
                if hasattr(imported_file, 'config'):
                    _check_config(imported_file.config, _overriding_parameters)
            else:
                print_errors(config + ' does not exist', do_exit=True)

        return config
