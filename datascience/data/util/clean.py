import pandas as pd

from engine.dataset.rasters.environmental_raster_glc import PatchExtractor

df = pd.read_csv('/home/data/grid_occs.csv', header='infer', sep=';', low_memory=False)
df = df[['Latitude', 'Longitude', 'id']]  # Pour Benjamin
print(df['Latitude'].size)

extractor = PatchExtractor('/home/data/rasters_GLC19/', size=64, verbose=True)
extractor.add_all()

keep = [True] * df['Latitude'].size

for idx, row in enumerate(df.iterrows()):
    lat, lon = row[1]['Latitude'], row[1]['Longitude']
    try:
        p = extractor[lat, lon]
        if not (p.shape[1] == p.shape[2] == 64):
            print('\tRemoving data point')
            keep[idx] = False
    except ValueError:
        print('\tRemoving data point')
        keep[idx] = False
    if (idx + 1) % 10000 == 0:
        print(idx)

df = df[keep]
print(df['Latitude'].size)
df.to_csv('/home/data/grid_occs_cleaned.csv', sep=';', header=True, index=False)
