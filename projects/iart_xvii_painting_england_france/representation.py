import torch

from datascience.data.loader import PaintingDatasetGenerator
from datascience.ml.metrics import ValidationAccuracy
from datascience.ml.neural.checkpoints import create_model
from datascience.ml.neural.models.pretrained import initialize_model
from datascience.ml.neural.representation.representation import extract_representation
from datascience.ml.neural.supervised import fit

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

train, _, _ = generator.painter_dataset(test_size=0., val_size=0.)
representation_dataset, _, _ = generator.country_dataset_one_fold(painter_val=None, painter_test=None)

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


def do_extraction(fig_name='representation_tsne'):
    representation, colors = extract_representation(representation_dataset, model)
    print(representation.shape, colors.shape)
    representation_embedded = TSNE(n_components=2).fit_transform(representation)

    ax = plt(fig_name).gca()

    ax.scatter(representation_embedded[:, 0], representation_embedded[:, 1], c=colors)

    ax.set_title('Painting representation')


do_extraction()

stats = fit(
    model, train=train, test=train, training_params=training_params, validation_params=validation_params,
    optim_params=optim_params, cross_validation_params=cross_validation_params
)

do_extraction(fig_name='representation_tsne_final')

save_fig()
