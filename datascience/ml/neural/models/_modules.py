from datascience.ml.neural.models.util import one_input, one_label, load_or_create
from engine.core import module
from engine.parameters import special_parameters
from engine.logging.logs import print_info


@module
def load_create_nn(model_class, from_scratch=None, model_params={}, p_input=one_input, p_label=one_label):
    if from_scratch is None and hasattr(special_parameters, 'from_scratch'):
        from_scratch = special_parameters.from_scratch
    return load_or_create(model_class, from_scratch, model_params, p_input=p_input, p_label=p_label)


@module
def print_model_parameters(model):
    for name, param in model.named_parameters():
        print_info(name + ' ' + str(param.shape))
    print_info('\n' + '*' * 50 + '\n')
    for name, param in model.named_parameters():
        print_info(name + ' *' * 10 + '\n' + str(param.data.detach().numpy()).replace('array', ''))
