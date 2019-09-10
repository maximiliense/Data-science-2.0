import matplotlib.pyplot as plt
from pyproj import Proj, Transformer
import pandas as pd

proj_in = Proj(init='epsg:4326')
proj_out = Proj(init='epsg:3035')
transformer = Transformer.from_proj(proj_in, proj_out)


def project(longitude, latitude):
    x, y = transformer.transform(latitude, longitude)
    return x / 1000, y / 1000


def execute():
    df_train = pd.read_csv("/home/bdeneu/data/occurrences_glc18.csv", header='infer', sep=';', low_memory=False)
    df_test = pd.read_csv("/home/bdeneu/data/occurrences_glc18_test_withlabel.csv", header='infer', sep=';', low_memory=False)
    d_train = df_train[['Latitude', 'Longitude']].to_numpy()
    d_test = df_test[['Latitude', 'Longitude']].to_numpy()

    geo_tr = project(d_train[:, 0], d_train[:, 1])
    geo_te = project(d_test[:, 0], d_test[:, 1])

    print(geo_te)
    s = 0.8
    plt.style.use('classic')
    fig, ax = plt.subplots()
    ax.scatter(geo_tr[0][:], geo_tr[1][:], color='#00c000', marker='s', s=s)
    # ax.scatter(geo_te[0][:], geo_te[1][:], color='r', marker='s', s=s, label="test")
    # ax = fig.add_subplot(111, axisbg='white')

    ax.set_xlim(3200, 4400)
    ax.set_ylim(2000, 3200)
    ax.spines['bottom'].set_color('#dddddd')
    ax.spines['top'].set_color('#dddddd')
    ax.spines['right'].set_color('#dddddd')
    ax.spines['left'].set_color('#dddddd')
    ax.tick_params(axis='x', colors='#dddddd')
    ax.tick_params(axis='y', colors='#dddddd')
    ax.yaxis.label.set_color('#dddddd')
    ax.xaxis.label.set_color('#dddddd')
    ax.title.set_color('#dddddd')
    print("here")
    plt.show()
    fig.savefig('occurrences_glc18.png', transparent=True)


if __name__ == "__main__":
    execute()
