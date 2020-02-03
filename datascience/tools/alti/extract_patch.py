from datascience.tools.alti.alti_image import TileManager
import pandas as pd

# from datascience.data.util.source_management import check_source


def extract_patch(source_tiles, source_occs, offset=0, check_file=True):
    """
    Extract IGN patch from IGN maps.
    :param source:
    :param offset:
    :param check_file:
    :return:
    """

    # checking the source
    # r = check_source(source_tiles)
    # occs = check_source(source_occs)


    # extract manager
    # t_manager = TileManager(r['tiles'])
    t_manager = TileManager("/home/benjamin/alti/")
    extract_size = 256
    extract_res = 1.0

    # loading the occurrence file
    df = pd.read_csv("/home/benjamin/occurrences.csv", header='infer', sep=';', low_memory=False)

    # sorting the dataset to optimise the extraction
    df.sort_values('lat', inplace=True)

    # offset management
    df = df.iloc[offset:]

    print(str(len(df)) + ' occurrences to extract!')

    t_manager.extract_patches(
        # df[[occs['longitude'], occs['latitude'], occs['id_name']]], r['patches'], extract_res, extract_size, check_file=check_file
        df[['lon', 'lat', 'id']], "/home/data/GLC/alti_1m/", extract_res, extract_size, check_file=check_file
    )
