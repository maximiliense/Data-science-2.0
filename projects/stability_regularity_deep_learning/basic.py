from datascience.data.loader import cifar10, mnist
from datascience.ml.metrics import ValidationAccuracy
from datascience.ml.neural.supervised import fit
from datascience.ml.neural.models.cnn import CustomizableCNN
from datascience.ml.neural.supervised.callbacks.callbacks import NewStatCallback
from datascience.ml.neural.supervised.train.checkpoints import create_model
from engine.parameters import get_parameters

# load MNIST or CIFAR10
if get_parameters('mnist', False):
    train, test = mnist()
else:
    # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    train, test = cifar10()

model_params = {
    'im_shape': train[0][0].shape,
    'conv_layers': (150, 150),
    'linear_layers': (128, 128)
}

model = create_model(model_class=CustomizableCNN, model_params=model_params)

training_params = {
    'lr': 0.01,
    'iterations': [50, 80, 100],  # iterations with learning rate decay
    'log_modulo': -1,  # print loss once per epoch
    'val_modulo': 1,  # run a validation on the validation set every 5 epochs
    'batch_size': 1024

}

validation_params = {
    'metrics': (ValidationAccuracy(1),),
    'vcallback': (NewStatCallback(train),)
}

fit(
    model, train, test, training_params=training_params, validation_params=validation_params
)
