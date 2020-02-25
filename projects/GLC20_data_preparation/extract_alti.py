from datascience.tools.alti.extract_patch import extract_patch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("occs", help="occurrences file")
args = parser.parse_args()
occs_file = args.occs


extract_patch(source_tiles="/home/benjamin/alti/", occs_file=occs_file, dest="/home/data/GLC/alti_1m_us/", check_file=True)
