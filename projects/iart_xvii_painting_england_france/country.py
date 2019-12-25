from datascience.data.loader import PaintingDatasetGenerator
from datascience.ml.neural.supervised import fit
from datascience.ml.metrics import ValidationAccuracy
from datascience.ml.neural.checkpoints import create_model
from datascience.ml.neural.models.pretrained import initialize_model

import torch

from engine.logging import print_h1
from engine.path import output_path

model_params = {
    # for inception, aux_logits must be False
    'model_name': 'inception',
    'num_classes': 2,
    'feature_extract': True
}
input_size = 299  # inception
generator = PaintingDatasetGenerator(source='paintings_xviii')

export_result = output_path('result.txt')

painter_list = generator.unique_painters()
for i in range(len(painter_list)):
    painter_val = painter_list[i]
    painter_test = painter_list[(i+1) % len(painter_list)]
    print_h1('||| PAINTER VAL: ' + painter_val + ', PAINTER TEST: ' + painter_test + ' |||')
    with open(export_result, 'a') as f:
        f.write('||| PAINTER VAL: ' + painter_val + ', PAINTER TEST: ' + painter_test + ' |||' + '\n')
    train, val, test = generator.country_dataset_one_fold(painter_val=painter_val, painter_test=painter_test)

    model = create_model(model_class=initialize_model, model_params=model_params)
    mmodel = model.module if type(model) is torch.nn.DataParallel else model
    mmodel.aux_logits = False
    input_size = mmodel.input_size

    training_params = {
        'iterations': [100, 130, 150, 160],
        'batch_size': 256,
    }

    optim_params = {
        'lr': 0.001
    }

    validation_params = {
        'metrics': (ValidationAccuracy(1),)
    }

    cross_validation_params = {
        'cross_validation': True,
        'min_epochs': 50
    }

    stats = fit(
        model, train=train, val=val, test=test, training_params=training_params, validation_params=validation_params,
        optim_params=optim_params, cross_validation_params=cross_validation_params
    )
    with open(export_result, 'a') as f:
        f.write(str(stats) + '\n\n')
    del stats
    del model
