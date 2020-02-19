from torch.utils.data import Dataset
import os
import torch

import numpy as np

from datascience.data.rasters.environmental_raster_glc2020 import PatchExtractor

_mapping_ces_bio = {
    
}

patch_extractor = None


class DatasetGLC20(Dataset):
    def __init__(self, labels, dataset, ids, rasters, patches, mapping_ces_bio=True):
        """
        :param labels:
        :param dataset: (latitude, longitude, ID)
        :param mapping_ces_bio:
        """
        self.mapping_ces_bio = mapping_ces_bio
        self.labels = labels
        self.dataset = dataset
        self.ids = ids

        self.one_hot_size = 18
        self.one_hot = np.eye(self.one_hot_size)

        self.rasters = rasters

        self.patches = patches
        global patch_extractor
        if patch_extractor is None:

            # 256 is mandatory as images have been extracted in 256 and will be stacked in the __getitem__ method
            patch_extractor = PatchExtractor(rasters, size=256, verbose=True)
            patch_extractor.add_all()

        self.extractor = patch_extractor

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        latitude = self.dataset[idx][0]
        longitude = self.dataset[idx][1]
        id_ = str(self.ids[idx])

        # folders that contain patches
        folder_1 = id_[-2:]
        folder_2 = id_[-4:-2]

        # path to patches
        path = os.path.join(self.patches, folder_1, folder_2, id_)
        path_alti = path + '_alti.npy'
        path_rgb_ir_lc = path + '_rgb_ir_lc.npy'

        # extracting patch from rasters
        tensor = self.extractor[(latitude, longitude)]

        # extracting altitude patch
        alti = np.load(path_alti)

        # extracting rgb infra-red and land cover
        rgb_ir_lc = np.load(path_rgb_ir_lc)

        # transforming landcover in one hot encoding
        lc = rgb_ir_lc[4:5]
        lc_reshaped = lc.reshape((lc.shape[1] * lc.shape[2],))
        lc_one_hot = self.one_hot[lc_reshaped]
        lc_one_hot = lc_one_hot.reshape((self.one_hot_size, lc.shape[1], lc.shape[2]))

        # concatenating all patches
        tensor = np.concatenate((tensor, alti.reshape((1, alti.shape[0], alti.shape[1])), rgb_ir_lc[:4], lc_one_hot))

        return torch.from_numpy(tensor).float(), self.labels[idx]
