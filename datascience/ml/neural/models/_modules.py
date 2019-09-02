from engine.core import module
from engine.logging import print_info


@module
def print_model_parameters(model):
    for name, param in model.named_parameters():
        print_info(name + ' ' + str(param.shape))
    print_info('\n' + '*' * 50 + '\n')
    for name, param in model.named_parameters():
        print_info(name + ' *' * 10 + '\n' + str(param.data.detach().numpy()).replace('array', ''))
