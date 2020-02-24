from datascience.tools.alti.extract_patch import extract_patch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("occs", help="occurrences file")
args = parser.parse_args()
occs_file = args.occs


extract_patch(source_tiles="alti", occs_file=occs_file, check_file=True)
