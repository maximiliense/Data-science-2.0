"""
This code extract the IGN archives and export the results into patches

usage: sjobs projects/max_env/extract_ign.py  #  for 5m patch
       sjobs projects/max_env/extract_ign.py -m source50cm=True  # for 50cm patch
"""

from datascience.tools.ign.check_extraction import check_extraction
from datascience.tools.ign.extract_7z import extract_7z
from datascience.tools.ign.extract_patch import extract_patch
from engine.parameters.special_parameters import get_parameters

if get_parameters('test', False):
    test = '_test'
else:
    test = ''
if get_parameters('source50cm', False):
    source = 'ign_50cm_maps_and_patches' + test
else:
    source = 'ign_5m_maps_and_patches' + test

if get_parameters('check_only', False):
    check_extraction(source=source)
else:
    if get_parameters('uncompress', False):
        # uncompress the IGN maps
        extract_7z(source=source)

    # extract patches from a dataset and the IGN maps
    extract_patch(source, offset=get_parameters('offset', 0))

    # check extraction, save errors and filter dataset
    check_extraction(source=source)
