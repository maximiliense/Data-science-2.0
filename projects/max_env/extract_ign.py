from datascience.tools.ign.check_extraction import check_extraction
from datascience.tools.ign.extract_7z import extract_7z
from datascience.tools.ign.extract_patch import extract_patch
from engine.parameters.special_parameters import get_parameters


if get_parameters('check_only', False):
    check_extraction(source='full_ign')
else:
    if get_parameters('uncompress', True):
        # uncompress the IGN maps
        extract_7z()

    # extract patches from a dataset and the IGN maps
    extract_patch(get_parameters('offset', 0))

    # check extraction, save errors and filter dataset
    check_extraction(source='full_ign')
