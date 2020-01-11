from datascience.data.loader import cifar10, mnist
from datascience.ml.metrics import ValidationAccuracy
from datascience.ml.neural.supervised import fit
from datascience.ml.neural.models.cnn import CustomizableCNN
# from datascience.ml.neural.supervised.callbacks import NewStatCallback
from datascience.ml.neural.checkpoints import create_model
from datascience.ml.neural.supervised.callbacks.callbacks import FilterVarianceCallback, \
    AlignmentMetricCallback  # ParameterVarianceCallback,
from engine.parameters import get_parameters

import torch

# load MNIST or CIFAR10
train, test = mnist() if get_parameters('mnist', False) else cifar10()
# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

model_params = {
    'im_shape': train[0][0].shape,
    'conv_layers': (64,),  # (150, 150),
    'linear_layers': tuple(),  # (128, 128),
    'pooling': torch.nn.AvgPool2d,
    'conv_size': 3
}

model = create_model(model_class=CustomizableCNN, model_params=model_params)

training_params = {
    'iterations': [120],  # iterations with learning rate decay
    'log_modulo': -1,  # print loss once per epoch
    'val_modulo': 1,  # run a validation on the validation set every 5 epochs
    'batch_size': 512

}

optim_params = {
    'lr': 0.01,
    'momentum': 0.0
}

validation_params = {
    'metrics': (ValidationAccuracy(1),),
    'vcallback': (FilterVarianceCallback(averaged=False, window_size=10),)  # (AlignmentMetricCallback(),NewStatCallback(train),)
}

fit(
    model, train, test, training_params=training_params, validation_params=validation_params, optim_params=optim_params
)
