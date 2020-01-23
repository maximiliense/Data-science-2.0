from datascience.data.loader import PaintingDatasetGenerator
from datascience.ml.neural.checkpoints import create_model
from datascience.ml.neural.models.pretrained import initialize_model
from datascience.visu.deep.input_importance import plot_input_importance

model_params = {
    # for inception, aux_logits must be False
    'model_name': 'inception',
    'num_classes': 2,
    'feature_extract': True
}

model = create_model(model_class=initialize_model, model_params=model_params)

generator = PaintingDatasetGenerator(source='paintings_xviii')

dataset, _, _, _ = generator.painter_dataset(0, 0)

for i in range(50):
    plot_input_importance(model, dataset, i)
