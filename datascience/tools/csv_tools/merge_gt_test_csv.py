import pandas as pd
import numpy as np

df_test = pd.read_csv('/home/bdeneu/Downloads/occurrences_test.csv', header='infer', sep=';', low_memory=False)
df_gt = pd.read_csv('/home/bdeneu/Downloads/gt_file.csv', header=None, sep=';', low_memory=False)

print(df_test)
print(df_gt)

df_gt.columns=['patch_id', 'species_glc_id']

print(df_gt)

sr = df_gt['species_glc_id']

print(sr)
print(type(sr))

result = pd.concat([df_test, sr], axis=1)

result.to_csv(index=False, sep=';', path_or_buf='/home/bdeneu/Downloads/occurrences_glc18_test_groundtrust.csv')
