import torch.optim as optimizer
from datascience.data.loader.cifar10 import cifar
from datascience.ml.metrics import ValidationAccuracy
from datascience.ml.neural.supervised import fit
from datascience.ml.neural.models import CNN, load_create_nn

from engine.tensorboard import add_rgb_grid_stack, add_graph

train, test = cifar()

images = add_rgb_grid_stack('grid', train)

# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

model = load_create_nn(model_class=CNN)

add_graph(model, images)

training_params = {
    'lr': 0.1,
    'iterations': [50, 80, 100],  # iterations with learning rate decay
    'log_modulo': -1,
    'val_modulo': 5,
    'batch_size': 10

}

validation_params = {
    'metrics': (ValidationAccuracy(1),)
}

optim_params = {
    'momentum': 0.0,
    'optimizer': optimizer.SGD
}

fit(
    model, train, test, training_params=training_params, validation_params=validation_params, optim_params=optim_params
)
