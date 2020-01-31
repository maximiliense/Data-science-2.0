import ast

with open("/home/bdeneu/Desktop/lewa.json", 'r') as f:
    s = f.read()
    liste = ast.literal_eval(s)

for i, el in enumerate(liste):
    # print("{}\t{}\tcircle1\tred".format(el["lat"], el["lon"]))
    print("{}, {} <green-dot>".format(el["lat"], el["lon"]))
