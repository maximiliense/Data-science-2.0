from datascience.data.loader.cifar10 import cifar
from datascience.ml.metrics import ValidationAccuracy
from datascience.ml.neural.supervised import fit
from datascience.ml.neural.models import CNN, load_create_nn

train, test = cifar()

# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

model_params = {
    'relu': True,
    'width': 100  # 5, 20, 50, 100, 200
}

model = load_create_nn(model_class=CNN, model_params=model_params)

training_params = {
    'lr': 0.1,
    'iterations': [50, 80, 100],  # iterations with learning rate decay
    'log_modulo': -1,  # print loss once per epoch
    'val_modulo': 5,  # run a validation on the validation set every 5 epochs
    'batch_size': 10

}

validation_params = {
    'metrics': (ValidationAccuracy(1),)
}

fit(
    model, train, test, training_params=training_params, validation_params=validation_params
)
