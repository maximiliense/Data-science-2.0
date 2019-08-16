from datascience.data.loader.cifar10 import cifar
from datascience.ml.metrics import ValidationAccuracy
from datascience.ml.neural.supervised import fit
from datascience.ml.neural.models import load_create_nn
from datascience.ml.neural.models.cnn import CustomizableCNN
from datascience.ml.neural.supervised.callbacks.callbacks import NewStatCallback

train, test = cifar()

# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

model_params = {
}

model = load_create_nn(model_class=CustomizableCNN, model_params=model_params)

training_params = {
    'lr': 0.01,
    'iterations': [50, 80, 100],  # iterations with learning rate decay
    'log_modulo': -1,  # print loss once per epoch
    'val_modulo': 1,  # run a validation on the validation set every 5 epochs
    'batch_size': 1024

}

validation_params = {
    'metrics': (ValidationAccuracy(1),),
    'vcallback': (NewStatCallback(),)
}

fit(
    model, train, test, training_params=training_params, validation_params=validation_params
)
