import pandas as pd
import ast

df = pd.read_csv("/home/data/occurrences_gbif_taxref.csv", header='infer', sep=';', low_memory=False)

with open("/home/benjamin/pycharm/Data-science-2.0/projects/ecography/inception_rs_normal_do07_4_20190831035247/index.json", 'r') as f:
    s = f.read()
    dict_label = ast.literal_eval(s)

inv_dict_label = {v: k for k, v in dict_label.items()}

for row in df.iterrows():
    row[1]['Label'] = inv_dict_label[row[1]['Label']]

df.to_csv(path_or_buf="/home/benjamin/full_dataset.csv", sep=';', index=False)
