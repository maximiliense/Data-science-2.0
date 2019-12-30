import torch
from matplotlib import cm

from datascience.data.loader import PaintingDatasetGenerator
from datascience.ml.metrics import ValidationAccuracy
from datascience.ml.neural.checkpoints import create_model
from datascience.ml.neural.models.pretrained import initialize_model
from datascience.ml.neural.representation.representation import extract_representation
from datascience.ml.neural.supervised import fit

import numpy as np

from sklearn.manifold import TSNE

from datascience.visu.util import plt, save_fig

model_params = {
    # for inception, aux_logits must be False
    'model_name': 'inception',
    'num_classes': 60,
    'feature_extract': False
}

model = create_model(model_class=initialize_model, model_params=model_params)
mmodel = model.module if type(model) is torch.nn.DataParallel else model
mmodel.aux_logits = False
input_size = mmodel.input_size

generator = PaintingDatasetGenerator(source='paintings_xviii')

train, _, _, index = generator.painter_dataset(test_size=0., val_size=0.)
representation_dataset, _, _, index_country = generator.country_dataset_one_fold(painter_val=None, painter_test=None)
training_params = {
    'iterations': [100, 130, 150, 160],
    'batch_size': 64,
}

optim_params = {
    'lr': 0.001
}

# only 2 classes
validation_params = {
    'metrics': (ValidationAccuracy(1),)
}

cross_validation_params = {
    'cross_validation': True,
    'min_epochs': 50
}


def clean_labels(label):
    if label == 'french_painter':
        return 'French'
    elif label == 'english_painter':
        return 'English'
    else:
        return label


def do_extraction(dataset, labels_index, fig_name='representation_tsne'):
    representation, colors, labels = extract_representation(dataset, model, labels_index=labels_index)
    count_labels = {k: True for k in labels}

    representation_embedded = TSNE(n_components=2).fit_transform(representation)

    zipped = list(zip(representation_embedded, colors, labels))
    zipped.sort(key=lambda tup: tup[2])
    c = zipped[0][2]
    artists = []
    col, rep = [], []

    artists.append((rep, col, c))
    colors = cm.rainbow(np.linspace(0, 1, len(count_labels)))
    for row in zipped:
        if row[2] != c:
            col, rep = [], []
            c = row[2]
            artists.append((rep, col, c))
        col.append(colors[row[1]])
        rep.append(row[0])

    ax = plt(fig_name, figsize=(8, 10)).gca()

    for row in artists:
        col = np.array(row[1])
        rep = np.array(row[0])
        ax.scatter(rep[:, 0], rep[:, 1], c=col, label=clean_labels(row[2]))

    ax.legend(bbox_to_anchor=(0, 0, 1, 1), bbox_transform=plt(fig_name).gcf().transFigure, prop={'size': 6},
              ncol=4, loc=3)
    ax.set_title('Painting representation')


do_extraction(representation_dataset, index_country, fig_name='representation_tsne_country')

stats = fit(
    model, train=train, test=train, training_params=training_params, validation_params=validation_params,
    optim_params=optim_params, cross_validation_params=cross_validation_params
)

do_extraction(train, index, fig_name='representation_tsne_final_painter')

do_extraction(representation_dataset, index_country, fig_name='representation_tsne_final_country')

save_fig()
