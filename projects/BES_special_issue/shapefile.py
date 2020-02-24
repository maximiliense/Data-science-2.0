import shapefile as shapefile_import
from shapely.geometry import Polygon, Point
import random
import numpy as np
from pyproj import Proj, Transformer
import csv
import ast
import fiona

shape = fiona.open("/home/bdeneu/Desktop/BES_special_issue/zone_d'étude_ramieres/Etude_LPO_2017.shp")
print(shape.schema)
pol = shape.next()
print(pol)

with open("/home/bdeneu/metropole.geojson") as geoj:
    string = geoj.readlines()
    france = Polygon(ast.literal_eval(string))

open_type = 'w'

inProj = Proj(init='epsg:2154')  # lamb93
outProj = Proj(init='epsg:4326')  # wgs84

# self.transformer_in_out = Transformer.from_proj(in_proj, out_proj)
transformer = Transformer.from_proj(outProj, inProj)

with open("/home/bdeneu/Desktop/allex.json", 'r') as f:
    s = f.read()
    liste = ast.literal_eval(s)

for el in liste:
    polygon = pol['geometry']['coordinates'][0]
    poly = Polygon(polygon)

    #print(polygon)
    x, y = transformer.transform(el["lon"], el["lat"])

    point = Point(x, y)
    # point in polygon test
    if poly.contains(point):
        print("{}, {} <blue-dot>".format(el["lat"], el["lon"]))

"""
for el in liste:
    shpfilePoints = []
    rep = 0
    polygon = r.shapes()
    for i, shape in enumerate(polygon):
        shpfilePoints = shape.points
        polygon = shpfilePoints
        poly = Polygon(polygon)

        #print(polygon)
        x, y = Transformer.transform(el["lon"], el["lat"])

        point = Point(x, y)
        # point in polygon test
        if poly.contains(point):
            print("{}, {} <green-dot>".format(el["lat"], el["lon"]))
"""
