from datascience.data.loader import occurrence_loader
from datascience.data.datasets import EnvironmentalIGNDataset
from datascience.visu.util import save_fig
from datascience.visu.patch import pplot_patch
import numpy as np


# with option --more idx=12 to change the index from the command line...
from engine.logging import print_info
from engine.parameters.special_parameters import get_parameters


# load the idx + 1 first elements

idx = get_parameters('idx', 0)

train, _, _ = occurrence_loader(
    EnvironmentalIGNDataset,
    source='full_ign',
    id_name='X_key',
    label_name='glc19SpId',
    validation_size=0,
    test_size=0,
    limit=idx + 1
)

patch, _ = train[idx]

patch = [l.int() for l in patch]

patch = patch[:-3] + [np.transpose(np.stack(patch[-3:], axis=0), (1, 2, 0))]

print_info('Printing patch at ' + str(train.dataset[idx]))

pplot_patch(patch, header=train.named_dimensions)

save_fig()
