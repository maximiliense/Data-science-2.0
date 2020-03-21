import numpy as np
import torch
from torch.nn import DataParallel

from datascience.data.datasets.vp_dataset import VisualPatternDataset
from datascience.ml.metrics import ValidationAccuracy
from datascience.ml.neural.checkpoints import create_model
from datascience.ml.neural.models.cnn import CustomizableCNN
from datascience.ml.neural.supervised import fit
from datascience.ml.neural.supervised.callbacks.callbacks import FilterVarianceCallback
from datascience.visu.util import plt, save_fig
from matplotlib import cm

pattern_1 = np.array([[1, 0], [0, 1]])
pattern_2 = np.array([[0, 0], [0, 0]])

patterns_list = [pattern_1, pattern_2]

for i, p in enumerate(patterns_list):
    ax = plt('pattern_' + str(i)).gca()
    ax.imshow(p, cmap=cm.Greys_r, extent=(-3, 3, 3, -3))
    ax.axis('off')


trainset = VisualPatternDataset(1000, [pattern_1, pattern_2], [[0.4, 0.], [0., 0.4]], 0.5)
testset = VisualPatternDataset(500, [pattern_1, pattern_2], [[0.4, 0.], [0., 0.4]], 0.5)

for i in range(10):
    ax = plt('image_' + str(trainset[i][1]) + '_' + str(i)).gca()
    ax.imshow(trainset[i][0].squeeze().cpu().numpy(), cmap=cm.Greys_r, extent=(-3, 3, 3, -3))
    ax.axis('off')

model_params = {
    'im_shape': trainset[0][0].shape,
    'conv_layers': (8,),  # (150, 150),
    'linear_layers': tuple(),  # (128, 128),
    'pooling': torch.nn.AvgPool2d,
    'conv_size': 2
}

model = create_model(model_class=CustomizableCNN, model_params=model_params)

training_params = {
    'iterations': [40, 50],  # iterations with learning rate decay
    'log_modulo': -1,  # print loss once per epoch
    'val_modulo': 1,  # run a validation on the validation set every 5 epochs
    'batch_size': 16

}

optim_params = {
    'lr': 0.1,
    'momentum': 0.0
}

validation_params = {
    'metrics': (ValidationAccuracy(1),),
    'vcallback': (FilterVarianceCallback(averaged=False, window_size=10),)
}

fit(
    model, trainset, testset, training_params=training_params, validation_params=validation_params,
    optim_params=optim_params
)

if type(model) is DataParallel:
    m = model.module
else:
    m = model

convolutions = m.conv_layers[0][0].weight.detach().cpu().numpy()

nb_plots = min(10, convolutions.shape[0])

for i in range(nb_plots):
    ax = plt('filters_' + str(i)).gca()
    ax.imshow(convolutions[i][0], cmap=cm.Greys_r, extent=(-3, 3, 3, -3))
    ax.axis('off')
save_fig()
