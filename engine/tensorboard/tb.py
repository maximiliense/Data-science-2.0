from torch.utils.tensorboard import SummaryWriter
from engine.parameters import special_parameters

from engine.logging.logs import print_info

import torch
import torchvision

import os

image_step = {}
scalar_step = {}


def initialize_tensorboard():
    print_info('Initializing tensorboard at {}'.format(special_parameters.tensorboard_path))
    special_parameters.tensorboard_writer = SummaryWriter(
        log_dir=os.path.join(special_parameters.tensorboard_path, special_parameters.experiment_name)
    )


def add_image(tag, img_tensor, walltime=None):
    if special_parameters.tensorboard_writer is not None:
        global image_step
        if 'tag' not in image_step:
            image_step[tag] = 0
        special_parameters.tensorboard_writer.add_image(
            tag, img_tensor, global_step=image_step[tag], walltime=walltime
        )
        image_step[tag] += 1


def add_graph(model, input_to_model):
    if special_parameters.tensorboard_writer is not None:
        special_parameters.tensorboard_writer.add_graph(model, input_to_model)  # , verbose=True)
        special_parameters.tensorboard_writer.flush()


def add_scalar(tag, scalar_value, walltime=None):
    if special_parameters.tensorboard_writer is not None:
        if tag not in scalar_step:
            scalar_step[tag] = 0
        special_parameters.tensorboard_writer.add_scalar(
            tag, scalar_value, global_step=scalar_step[tag], walltime=walltime
        )
        scalar_step[tag] += 1


def add_rgb_grid(tag, images):
    if special_parameters.tensorboard_writer is not None:
        grid = torchvision.utils.make_grid(images)
        add_image(tag, grid)


def add_rgb_grid_stack(tag, dataset):
    if special_parameters.tensorboard_writer is not None:
        r = torch.stack(tuple(dataset[i][0] for i in range(10)), 0)
        add_rgb_grid(tag, r)
        return r
