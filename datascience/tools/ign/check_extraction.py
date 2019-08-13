from datascience.data.util.source_management import check_source

import pandas as pd
import progressbar
import os

from engine import print_errors, output_path, print_logs
from engine.core import module


@module
def check_extraction(source, save_errors=True, save_filtered=True, id_name='X_key'):
    """
    check if all patches from an occurrences file have been extracted. Can save the list of errors and
    filtered the dataset keeping the correctly extracted data.

    :param id_name: the column that contains the patch id that will be used to construct its path
    :param save_filtered: save the dataframe filtered from the error
    :param save_errors: save the errors found in a file
    :param source: the source referring the occurrence file and the patches path
    """

    # retrieve details of the source
    r = check_source(source)
    if 'occurrences' not in r or 'patches' not in r:
        print_errors('Only sources with occurrences and patches can be checked', do_exit=True)

    df = pd.read_csv(r['occurrences'], header='infer', sep=';', low_memory=False)
    nb_errors = 0
    errors = []
    for idx, row in progressbar.progressbar(enumerate(df.iterrows())):
        patch_id = str(int(row[1][id_name]))

        # constructing the path of a patch given its id
        path = os.path.join(r['patches'], patch_id[-2:], patch_id[-4:-2], patch_id + '.npy')

        # if the path does not correspond to a file, then it's an error
        if not os.path.isfile(path):
            errors.append(row[1][id_name])
            nb_errors += 1

    if nb_errors > 0:
        # summary of the error
        print_logs(str(nb_errors) + ' errors found during the check...')

        if save_errors:
            # filter the dataframe using the errors
            df_errors = df[df[id_name].isin(errors)]

            error_path = output_path('_errors.csv')
            print_logs('Saving error file at: ' + error_path)

            # save dataframe to the error file
            df_errors.to_csv(error_path, header=True, index=False, sep=';')
        if save_filtered:
            # filter the dataframe keeping the non errors
            df_filtered = df[~df[id_name].isin(errors)]
            filtered_path = r['occurrences'] + '.tmp'
            print_logs('Saving filtered dataset at: ' + filtered_path)
            df_filtered.to_csv(filtered_path, header=True, index=False, sep=';')
    else:
        print_logs('No error has been found!')
