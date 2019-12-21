from torchvision import transforms
import torch
from datascience.ml.metrics import ValidationAccuracy
from datascience.ml.neural.checkpoints import create_model
from datascience.ml.neural.models.pretrained import initialize_model
from datascience.data.loader import cifar10
from datascience.ml.neural.supervised import fit

model_params = {
    # for inception, aux_logits must be False
    'model_name': 'inception',
    'num_classes': 10,
    'feature_extract': True
}

model = create_model(model_class=initialize_model, model_params=model_params)

mmodel = model.module if type(model) is torch.nn.DataParallel else model
mmodel.aux_logits = False
input_size = mmodel.input_size

transform = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

train, test = cifar10(transform)

training_params = {
    'iterations': [50, 75, 100],
    'batch_size': 256,
}

optim_params = {
    'lr': 0.001
}

validation_params = {
    'metrics': (ValidationAccuracy(1),)
}

fit(
    model, train=train, test=test, training_params=training_params, validation_params=validation_params,
    optim_params=optim_params, cross_validation=True
)
