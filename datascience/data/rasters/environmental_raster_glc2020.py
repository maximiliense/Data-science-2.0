"""
"
"   Author: Maximilien Servajean - mservajean
"   Mail: servajean@lirmm.fr
"   Date: 04/01/2019
"
"   Description: The code to extract environmental tensors and environmental vectors given some environmental rasters.
"
"""
import numpy as np
import rasterio
import re
import warnings

from datascience.visu.util.util import plt, get_figure
from engine.logging.logs import print_debug, print_errors
from engine.parameters import special_parameters


# metadata used to setup some rasters
raster_metadata = {
    'bdticm': {'min_val': 0, 'max_val': 112467, 'nan': -2147483647, 'new_nan': -1},
    'bldfie': {'min_val': 93, 'max_val': 1828, 'nan': -32768, 'new_nan': 92},
    'cecsol': {'min_val': 0, 'max_val': 385, 'nan': -32768, 'new_nan': -1},
    'clyppt': {'min_val': 0, 'max_val': 81, 'nan': -32768, 'new_nan': -1},
    'orcdrc': {'min_val': 0, 'max_val': 524, 'nan': -32768, 'new_nan': -1},
    'phihox': {'min_val': 32, 'max_val': 98, 'nan': -32768, 'new_nan': 31},
    'sltppt': {'min_val': 0, 'max_val': 86, 'nan': -32768, 'new_nan': -1},
    'sndppt': {'min_val': 0, 'max_val': 99, 'nan': -32768, 'new_nan': -1},
}


class Raster(object):
    """
    Raster is dedicated to a single raster management...
    """
    def __init__(self, path, country='FR', normalized=False, transform=None, size=256, nan=None, new_nan=None, **kw):
        """
        Loads a tiff file describing an environmental raster into a numpy array and...

        :type new_nan:
        :param path: the path of the raster (the directory)
        :param nan: the value to use when NaN number are present. If False, then default values will be used
        :param normalized: if True the raster will be normalized (minus the mean and divided by std)
        :param transform: if a function is given, it will be applied on each patch.
        :param size: the size of a patch (size x size)
        """
        self.path = path
        self.no_data = nan
        self.normalized = normalized
        self.transform = transform
        self.size = size

        path = re.sub(r'/\/+/', '/', path)

        self.name = path.split('/')[-1] if path[-1] != '/' else path.split('/')[-2]
        print(path + '/' + self.name + '_' + country + '.tif')
        # src.meta
        # to avoid the annoying corresponding warning, temporary warning disabling...
        warnings.filterwarnings("ignore")

        src = rasterio.open(path + '/' + self.name + '_' + country + '.tif', nodata=nan)

        warnings.filterwarnings("default")

        if src.meta['crs'] is None:

            with open(path + '/' + 'GeoMetaData.csv') as f:
                metadata = f.read()

            m_split = metadata.split('\n')[1].split(';')

            # loading file data
            self.x_min = float(m_split[1])
            self.y_min = float(m_split[2])
            self.x_resolution = float(m_split[5])
            self.y_resolution = float(m_split[6])
            self.n_rows = int(m_split[3])
            self.n_cols = int(m_split[4])
        else:
            self.x_min = src.bounds.left
            self.y_min = src.bounds.bottom
            self.x_resolution = src.res[0]
            self.y_resolution = src.res[1]
            self.n_rows = src.height
            self.n_cols = src.width
        print(self.x_min, self.y_min, self.x_resolution, self.y_resolution, self.n_rows, self.n_cols)
        # some tiff do not contain geo data (stored in the file GeoMetaData)
        # loading the raster
        self.raster = np.squeeze(src.read())
        src.close()

        # value bellow min_value are considered incorrect and therefore no_data
        self.raster[self.raster == self.no_data] = new_nan
        self.raster[np.isnan(self.raster)] = new_nan

        if normalized:
            # normalizing the whole raster given available data (therefore avoiding no_data)...
            selected_cell = self.raster != nan
            self.raster[selected_cell] = (self.raster[selected_cell] - self.raster[selected_cell].mean()) \
                / self.raster[selected_cell].std()

        # setting the shape of the raster
        self.shape = self.raster.shape

    def _get_patch(self, item):
        """
        Avoid using this method directly

        :param item: the GPS position (latitude, longitude)
        :return: a patch
        """
        row_num = int(self.n_rows - (item[0] - self.y_min) / self.y_resolution)
        col_num = int((item[1] - self.x_min) / self.x_resolution)

        # environmental vector
        if self.size == 1:
            patch = self.raster[row_num, col_num].astype(np.float)

        else:
            half_size = int(self.size/2)
            patch = self.raster[
                    row_num-half_size:row_num+half_size,
                    col_num - half_size:col_num+half_size
                    ].astype(np.float)
        patch = patch[np.newaxis]
        return patch

    def __len__(self):
        """
        :return: the depth of the tensor/vector...
        """
        return 1

    def __getitem__(self, item):
        """
        The method to use to retrieve a patch.

        :param item: GPS position (latitude, longitude)
        :return: the extracted patch with eventually some transformations
        """
        # item is a tuple of (latitude, longitude)
        patch = self._get_patch(item)
        if self.transform:
            patch = self.transform(patch)

        return patch


class PatchExtractor(object):
    """
    PatchExtractor enables the extraction of an environmental tensor from multiple rasters given a GPS
    position.
    """
    def __init__(self, root_path, size=256, verbose=False, resolution=1.):
        self.root_path = root_path
        self.size = size

        self.verbose = verbose
        self.resolution = resolution

        self.rasters_fr = []
        self.rasters_us = []

    def add_all(self, normalized=False, transform=None):
        """
        Add all variables (rasters) available at root_path

        :param normalized: if True, each raster will be normalized
        :param transform: a function to apply on each patch
        """
        for key in sorted(raster_metadata.keys()):
            if 'ignore' not in raster_metadata[key]:
                self.append(key, normalized=normalized, transform=transform)

    def append(self, raster_name, **kwargs):
        """
        This method append a new raster given its name

        :param raster_name:
        :param kwargs: nan, normalized, transform
        """
        # you may want to add rasters one by one if specific configuration are required on a per raster
        # basis
        print_debug('Adding: ' + raster_name, end='')
        params = {**raster_metadata[raster_name]}
        for k in kwargs.keys():
            if kwargs[k] != 'default':
                params[k] = kwargs[k]
        try:
            r_us = Raster(self.root_path + '/' + raster_name, 'USA', size=self.size, **params)
            r_fr = Raster(self.root_path + '/' + raster_name, 'FR', size=self.size, **params)
            self.rasters_us.append(r_us)
            self.rasters_fr.append(r_fr)
            print_debug('')
        except rasterio.errors.RasterioIOError:
            print_errors(' (' + raster_name + ' not available...)')

    def clean(self):
        """
        Remove all rasters from the extractor.
        """
        print_debug('Removing all rasters...')
        self.rasters_fr = []
        self.rasters_us = []

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        str_ = ''

        def raster_str(r):
            result = ''
            result += '-' * 50 + '\n'
            result += 'title: ' + r.name + '\n'
            result += '\t x_min: ' + str(r.x_min) + '\n'
            result += '\t y_min: ' + str(r.y_min) + '\n'
            result += '\t x_resolution: ' + str(r.x_resolution) + '\n'
            result += '\t y_resolution: ' + str(r.y_resolution) + '\n'
            result += '\t n_rows: ' + str(r.n_rows) + '\n'
            result += '\t n_cols: ' + str(r.n_cols) + '\n'
            return result
        for r in self.rasters_fr:
            str_ += raster_str(r)
        for r in self.rasters_us:
            str_ += raster_str(r)

        return str_

    def __getitem__(self, item):
        """
        :param item: the GPS location (latitude, longitude)
        :return: return the environmental tensor or vector (size>1 or size=1)
        """
        rasters = self._raster(item)
        if len(rasters) > 1:
            return np.concatenate([r.__getitem__(item) for r in rasters])
        else:
            return np.array([rasters[0].__getitem__(item)])

    def __len__(self):
        """
        :return: the number of variables (not the size of the tensor when some variables have a one hot encoding
                 representation)
        """
        return len(self.rasters_fr)

    def _raster(self, item):
        return self.rasters_fr if item[1] > -10. else self.rasters_us

    def plot(self, item, return_fig=False, style=special_parameters.plt_style, nb_cols=5, alpha=1.):
        """
        Plot an environmental tensor (size > 1)...

        :param alpha:
        :param nb_cols:
        :param item: the GPS location (latitude, longitude)
        :param return_fig: if True, the matplotlib fig will be returned, if False, it will be displayed
        :param style: style of the chart
        """

        if self.size > 1:
            rasters = self._raster(item)
            with plt().style.context(style):
                metadata = [
                    (r.name,
                     [
                         item[1] - self.size // 2 * r.x_resolution,
                         item[1] + self.size // 2 * r.x_resolution,
                         item[0] - self.size // 2 * r.y_resolution,
                         item[0] + self.size // 2 * r.y_resolution]
                     ) for r in rasters
                ]
                # metadata are the name of the variable and the bounding box in latitude-longitude coordinates

                # retrieve the patch... Eventually disabling the one hot encoding variables
                patch = self.__getitem__(item)

                # computing number of rows and columns...
                nb_rows = (patch.shape[0] + (nb_cols-1)) // nb_cols

                plt('patch', figsize=(nb_cols * 6.4 * self.resolution, nb_rows * 4.8 * self.resolution))
                fig = get_figure('patch')
                for k, i in zip(metadata, range(patch.shape[0])):
                    plt('patch').subplot(nb_rows, nb_cols, i + 1)
                    plt('patch').title(k[0], fontsize=20)
                    p = np.squeeze(patch[i])
                    plt('patch').imshow(p, extent=k[1], aspect='auto')
                    plt('patch').colorbar()
                fig.tight_layout()
                fig.patch.set_alpha(alpha)
            if return_fig:
                return fig
            else:
                fig.show()
                plt('patch').close(fig)
        else:
            raise ValueError('Plot works only for tensors: size must be > 1...')
