import matplotlib.pyplot as plt
from pyproj import Proj, Transformer
import numpy as np
from datascience.data.loader import occurrence_loader
from datascience.data.datasets import EnvironmentalDataset
from datascience.model_selection import SplitterGeoQuadra
from datascience.tools.occurrences_plot.occurrences_plot import plot_occurrences
from engine.parameters import get_parameters

source = get_parameters('source', 'glc18')
quad_size = get_parameters('quad_size', 0)
validation_size = get_parameters('validation_size', 0.1)
test_size = get_parameters('test_size', 0.1)

# loading dataset
train, val, test = occurrence_loader(EnvironmentalDataset, source=source, splitter=SplitterGeoQuadra(quad_size=quad_size), validation_size=validation_size, test_size=test_size)

plot_occurrences(train, val, test)