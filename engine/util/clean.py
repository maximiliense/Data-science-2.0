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

    torch_re = re.compile(r'{}.*\.torch$'.format(name))
    csv_re = re.compile(r'{}.*\.csv$'.format(name))
    image_re = re.compile(r'{}.*\.[jpegpnif]*$'.format(name))
    logs_re = re.compile(r'{}.*\.logs$'.format(name))
    txt_re = re.compile(r'{}.*\.txt$'.format(name))
    json_re = re.compile(r'{}.*\.json$'.format(name))
    pyc_re = re.compile(r'{}.*\.pyc$'.format(name))

    total = recursive_clean(path, (torch_re, csv_re, image_re, logs_re, txt_re, json_re, pyc_re), disp_only)

    print('Total amount: ' + _size(total))


def _size(memory_size):
    if memory_size > 1073741824:
        return str(memory_size >> 30) + ' MB'
    if memory_size > 1048576:
        return str(memory_size >> 20) + ' MB'
    elif memory_size > 1024:
        return str(memory_size >> 10) + ' KB'
    else:
        return str(memory_size) + ' B'


def recursive_clean(path, regex, disp_only=True):

    for file in os.listdir(path):
        path_file = os.path.join(path, file)
        if 'final' in path_file or 'keep' in path_file:
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

                print('rm ' + file_path + ' (' + _size(memory_size) + ')')

                if not disp_only:
                    os.remove(file_path)
