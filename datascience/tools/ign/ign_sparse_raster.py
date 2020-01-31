from pyproj import Proj

from engine.path import output_path
from datascience.tools.ign.ign_image import IGNImageManager
import pandas as pd
from datascience.data.util.source_management import check_source
from engine.core import module
from engine.logging import print_info


@module
def create_ign_sparse(source_occ, source_ign, patch_size=64, error_path=output_path("error_extract/"),
                      **kwargs):

    r = check_source(source_occ)
    occurrences = r['occurrences']
    r = check_source(source_ign)
    ign_images = r['maps']

    la93 = Proj(init='epsg:2154')

    # extract manager
    im_manager = IGNImageManager(ign_images)
    extract_size = patch_size
    extract_step = 1

    # loading the occurrence file
    df = pd.read_csv(occurrences, header='infer', sep=';', low_memory=False)
    max_lat = df['Latitude'].max()
    print(max_lat)


    # sorting the dataset to optimise the extraction
    df.sort_values('Latitude', inplace=True)

    print_info(str(len(df)) + ' occurrences to extract!')

