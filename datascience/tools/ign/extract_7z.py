import os
from engine.core import module


@module
def extract_7z(dir_name='/gpfsddn2/dataset/IGN/ign-url-list-5m/',
               dest_name='/gpfswork/rech/fqg/uid61lx/data/ign_maps/', extension='.7z'):

    os.chdir(dir_name)  # change directory from working dir to dir with files

    n = len(os.listdir(dir_name))

    for i, item in enumerate(os.listdir(dir_name)):  # loop through items in dir
        print("\n------------------------------------------------------------------------------")
        print(str(i+1)+"/"+str(n))
        if item.endswith(extension): # check for ".zip" extension
            file_name = os.path.abspath(item) # get full path of files
            print(file_name)
            print("\n")

            os.system('7z x '+file_name+' -o'+dest_name)
