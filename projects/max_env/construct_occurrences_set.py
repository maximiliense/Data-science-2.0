from datascience.data.util.source_management import check_source
import json
import pandas as pd

from engine.logging import print_info

source = check_source('glc20')
raw_occurrences_path = source['raw_source']
occurrences_path = source['occurrences']  # destination

with open(raw_occurrences_path, 'rb') as f:
    d = json.load(f)

data = {
    'id': [],
    'lat': [],
    'lon': [],
    'species_id': [],
    'species_name': []
}

for row in d:
    if row['results']['status'] == 'BEST_REF':
        data['id'].append(row['id'])
        data['lat'].append(row['lat'])
        data['lon'].append(row['lon'])
        data['species_id'].append(row['results']['id'])
        data['species_name'].append(row['results']['name'])

df = pd.DataFrame(data=data)

print_info('Saving file')

df.to_csv(occurrences_path, header=True, sep=';', index=False)
