import pandas as pd

df = pd.read_csv("/home/bdeneu/PL_trusted.csv", header='infer', sep=';', low_memory=False)

dic = {}

for row in df.iterrows():
    sp = row[1]['scName']
    la = row[1]['glc19SpId']
    if la not in dic:
        dic[la] = sp

with open("/home/bdeneu/label_name_PL_trusted.json", 'w') as f:
    f.write(str(dic))
