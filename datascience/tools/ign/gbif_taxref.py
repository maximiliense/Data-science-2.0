import pandas as pd


df_name = pd.read_csv("/home/bdeneu/label_name_valid.csv", header='infer', sep=';', low_memory=False)
df_occs = pd.read_csv("/home/bdeneu/occurrences_correct.csv", header='infer', sep=';', low_memory=False)

dict_new_names_labels = {"nan":"nan"}
dict_corres_labels = {}

for idx, row in enumerate(df_name.iterrows()):
    name = str(row[1]["validName"])
    if name not in dict_new_names_labels:
        dict_new_names_labels[name] = len(dict_new_names_labels)-1
    old_id = row[1]["id"]
    dict_corres_labels[old_id] = (dict_new_names_labels[name], name)

print("nb labels", len(dict_new_names_labels)-1)

list_lines = ["id;occurrenceID;Latitude;Longitude;Species;Label\n"]
n=0
for idx, row in enumerate(df_occs.iterrows()):
    old_label = row[1]["Label"]
    new_name = dict_corres_labels[row[1]["Label"]][1]

    if new_name != "nan":
        lon = row[1]["Longitude"]
        lat = row[1]["Latitude"]
        occ_id = str(row[1]["occurrenceID"])
        new_label = str(dict_corres_labels[row[1]["Label"]][0])
        id = str(len(list_lines))
        l = [id, occ_id, str(lat), str(lon), new_name, new_label]
        line = ";".join(l)+"\n"
        list_lines.append(line)
    else:
        n += 1

print("nb occurrences rejected", n)
with open("/home/bdeneu/occurrences_gbif_taxref.csv", "w") as file:
    file.writelines(list_lines)

dict_new_names_labels.pop('nan', None)
with open("/home/bdeneu/label_name_gbif_taxref_r.json", 'w') as f:
    f.write(str(dict_new_names_labels))
with open("/home/bdeneu/label_name_gbif_taxref.csv", 'w') as f:
    for name in dict_new_names_labels:
        f.write(str(dict_new_names_labels[name])+";"+name+"\n")
inv_dict = {v: k for k, v in dict_new_names_labels.items()}
with open("/home/bdeneu/label_name_gbif_taxref.json", 'w') as f:
    f.write(str(inv_dict))

