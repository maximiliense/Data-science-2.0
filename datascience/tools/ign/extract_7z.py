import os

from datascience.data.util.source_management import check_source
from engine.core import module


@module
def extract_7z(source, extension='.7z'):

    # loading a specific source
    r = check_source(source)

    dir_name = r['archive']
    dest_name = r['maps']

    os.chdir(dir_name)  # change directory from working dir to dir with files

    n = len(os.listdir(dir_name))

    for i, item in enumerate(os.listdir(dir_name)):  # loop through items in dir
        print("\n------------------------------------------------------------------------------")
        print(str(i+1)+"/"+str(n))
        if item.endswith(extension):  # check for ".zip" extension
            file_name = os.path.abspath(item)  # get full path of files
            print(file_name)
            print("\n")

            os.system('7z x '+file_name+' -o'+dest_name)
