from datascience.data.loader.paintings import PaintingDatasetGenerator
from datascience.ml.neural.supervised import fit
from datascience.ml.metrics import ValidationAccuracy
from datascience.ml.neural.checkpoints import create_model
from datascience.ml.neural.models.pretrained import initialize_model

import torch

model_params = {
    # for inception, aux_logits must be False
    'model_name': 'inception',
    'num_classes': 60,
    'feature_extract': True
}

model = create_model(model_class=initialize_model, model_params=model_params)
mmodel = model.module if type(model) is torch.nn.DataParallel else model
mmodel.aux_logits = False
input_size = mmodel.input_size

generator = PaintingDatasetGenerator(source='paintings_xviii')

train, val, test = generator.painter_dataset()  # change here for different experiments
print(train[0][0].size())

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
    model, train=train, val=val, test=test, training_params=training_params, validation_params=validation_params,
    optim_params=optim_params, cross_validation=True
)
