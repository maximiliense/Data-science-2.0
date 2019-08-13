import os
import re


def clean(path, name, disp_only=True):

    if name is None:
        name = '*'

    print('Cleaning files ' + name + ' from path: ' + path)
    if name == '*':
        name = '.*'
    else:
        name += '.*'

    model_re = re.compile(r'{}_model\.torch$'.format(name))
    model_tmp_re = re.compile(r'{}_[0-9]*_model\.torch$'.format(name))
    csv_re = re.compile(r'{}\.csv$'.format(name))
    loss_re = re.compile(r'{}\.[jpegpnif]*$'.format(name))
    lossl_re = re.compile(r'{}_loss\.logs$'.format(name))
    val_re = re.compile(r'{}_validation\.txt$'.format(name))
    index_re = re.compile(r'{}_index\.json$'.format(name))
    config_re = re.compile(r'{}config\.txt$'.format(name))
    pyc_re = re.compile(r'{}\.pyc$'.format(name))

    recursive_clean(path, (model_re, model_tmp_re, csv_re, loss_re, lossl_re, val_re, index_re, config_re,
                           pyc_re), disp_only)


def recursive_clean(path, regex, disp_only=True):

    for file in os.listdir(path):
        path_file = os.path.join(path, file)
        if 'final' in file:
            continue
        elif os.path.isdir(path_file):
            recursive_clean(path_file, regex, disp_only)
            if len(os.listdir(path_file)) == 0 and not disp_only:
                os.rmdir(path_file)
                print('rm ' + path_file)
        else:
            to_delete = False
            for r in regex:
                to_delete = to_delete or bool(re.search(r, path_file))

            if to_delete:
                file_path = os.path.join(path, file)
                memory_size = os.stat(file_path).st_size

                if memory_size > 1048576:
                    print('rm ' + file_path + ' (' + str(memory_size >> 20) + ' MB)')
                elif memory_size > 1024:
                    print('rm ' + file_path + ' (' + str(memory_size >> 10) + ' KB)')
                else:
                    print('rm ' + file_path + ' (' + str(memory_size) + ' B)')

                if not disp_only:
                    os.remove(file_path)
