import datascience.tools.ign.ign_image as igni
import pandas as pd

from datascience.data.util.source_management import check_source
from engine.core import module
from engine.logging import print_info


@module
def extract_patch(source, offset=0, check_file=True):
    """
    Extract IGN patch from IGN maps.
    :param source:
    :param offset:
    :param check_file:
    :return:
    """

    # checking the source
    r = check_source(source)

    # extract manager
    im_manager = igni.IGNImageManager(r['maps'])
    extract_size = 64
    extract_step = 1

    # loading the occurrence file
    df = pd.read_csv(r['occurrences'], header='infer', sep=';', low_memory=False)

    # offset management
    df = df.iloc[offset:]

    print_info(str(len(df)) + ' occurrences to extract!')

    # sorting the dataset to optimise the extraction
    df.sort_values('Latitude', inplace=True)

    im_manager.extract_patches(
        df[[r['longitude'], r['latitude']]],
        df[r['id_name']], r['patches'],
        size=extract_size,
        step=extract_step,
        check_file=check_file
    )
