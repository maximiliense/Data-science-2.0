from datascience.tools.ign.check_extraction import check_extraction
from datascience.tools.ign.extract_7z import extract_7z
from datascience.tools.ign.extract_patch import extract_patch
from engine.parameters.special_parameters import get_parameters


if get_parameters('source50cm', False):
    source = 'ign_50cm_maps_and_patches'
else:
    source = 'ign_5m_maps_and_patches'

if get_parameters('check_only', False):
    check_extraction(source=source)
else:
    if get_parameters('uncompress', True):
        # uncompress the IGN maps
        extract_7z(source=source)

    # extract patches from a dataset and the IGN maps
    extract_patch(source, offset=get_parameters('offset', 0))

    # check extraction, save errors and filter dataset
    check_extraction(source=source)
