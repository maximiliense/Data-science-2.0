import torch
import torchvision

from engine.tensorboard import add_rgb_grid, add_graph, add_scalar

# Writer will output to ./runs/ directory by default
from datascience.data.loader import mnist

import numpy as np

train_set, _ = mnist()
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
model = torchvision.models.resnet50(False)
# Have ResNet model take in grayscale rather than RGB
model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
images, labels = next(iter(train_loader))

add_rgb_grid('images', images)
print(model)
add_graph(model, images)

for n_iter in range(100):
    add_scalar('Loss/train', np.random.random(), n_iter)
    add_scalar('Loss/test', np.random.random(), n_iter)
    add_scalar('Accuracy/train', np.random.random(), n_iter)
    add_scalar('Accuracy/test', np.random.random(), n_iter)
