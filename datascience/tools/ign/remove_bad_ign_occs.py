import pandas as pd

df = pd.read_csv('/home/data/occurrences_GLC19/full_dataset.csv', sep=';', header='infer', low_memory=False)
df_errors = pd.read_csv('/home/data/ign_5m/error.csv', sep=';', header=1, low_memory=False)

to_remove = []

print(df_errors.columns)

for row in df_errors.iterrows():
    to_remove.append(row[1]['Occ_id'])
df = df[~df.X_key.isin(to_remove)]

df.to_csv('/home/data/occurrences_GLC19/full_dataset_ign.csv', sep=';', index=False)