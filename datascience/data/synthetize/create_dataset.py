from datascience.data.synthetize import synthetic_poly, circle
from engine.core import module


@module
def create_dataset(param_train=(100,), param_test=(100,), poly=True):
    """
    Create a dataset on a circle or on the x^3 function
    :param param_train: train parameters
    :param param_test: test parameters
    :param poly: x^3 if True, circle otherwise
    :return:
    """
    if poly:
        return synthetic_poly.Dataset(*param_train), synthetic_poly.Dataset(*param_train)
    else:
        return circle.Dataset(*param_train), circle.Dataset(*param_test)
