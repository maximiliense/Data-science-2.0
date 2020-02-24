from shapely.geometry import shape, Point
import pandas as pd
import geojson

print('new')
with open("/home/data/occurrences_GLC20/metropole.geojson") as f:
    gj = geojson.load(f)

polygon = shape(gj['geometry'])

df = pd.read_csv("/home/data/occurrences_GLC20/occurrences_GLC20_animals.csv", header='infer', sep=';', low_memory=False)
before_size = len(df.index)
print(before_size)

for index, row in df.iterrows():
    if index % 10000 == 9999:
        print(index + 1, before_size)
    point = Point(row['lon'], row['lat'])
    if not polygon.contains(point):
        df.drop(index, inplace=True)

print(len(df.index))
df.to_csv("/home/data/occurrences_GLC20/occurrences_GLC20_animals_filtered.csv", sep=";", index=False)
