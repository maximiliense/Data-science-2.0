from torchvision import transforms
import torch
import numpy as np

from torch.nn.functional import softmax

from datascience.visu.util import plt

import matplotlib.animation as animation

from engine.logging import print_info
from engine.path import output_path


def plot_input_importance(model, dataset, try_idx, dest='.'):
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    idx = try_idx
    image_transformed, _ = dataset[idx]
    image, _ = dataset.__getitem__(idx, transform=transform)
    image_transformed = image_transformed.unsqueeze(0)

    image_transformed.requires_grad = True

    output = softmax(model(image_transformed), dim=1)
    output[0][1].backward()
    gradient = torch.mean(torch.abs(image_transformed.grad.data.squeeze(0)), dim=0)
    nb_frames = 20
    quantiles = np.quantile(gradient, np.linspace(0.999999, 0.9, nb_frames))
    fps = []
    for it in range(nb_frames):
        mask = gradient > quantiles[it]

        transform_reshape = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image.size()[1], image.size()[2])),
            transforms.ToTensor()
        ])
        mask = transform_reshape(mask.int()).squeeze(0) > 0.5

        image_to_use = np.copy(image)
        for i in range(mask.size(0)):
            for j in range(mask.size(1)):
                if mask[i, j]:
                    image_to_use[0, i, j] = 1.
                    image_to_use[1, i, j] = 0.
                    image_to_use[2, i, j] = 0.
        fps.append(image_to_use.transpose((1, 2, 0)))
        print('appending...')
    fig = plt('test_figure2').gcf()
    ax = fig.gca()
    ax.axis('off')
    im = ax.imshow(fps[0])

    def animate(i):
        im.set_data(fps[i])
        return [im]

    anim = animation.FuncAnimation(
        fig,
        animate,
        frames=len(fps),
        interval=100,  # in ms
    )

    path = output_path(dest + '/test_anim_{}.gif'.format(try_idx))
    print_info(path)
    anim.save(path, dpi=80, writer='imagemagick')
