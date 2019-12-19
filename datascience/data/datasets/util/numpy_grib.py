import matplotlib
import numpy as np
from scipy import interpolate
import bisect

from datascience.visu.util import plt
from engine.parameters import special_parameters


def project(latitude, longitude, grib, scale=0.5):
    return int((latitude - grib.latitudes[0, 0]) / scale), int((longitude - grib.longitudes[0, 0]) / scale)


class NumpyGrib(object):

    def __init__(self, npz_path):
        """
        Constructor
        :param grbs_filename: original grib file
        :param uv_variables: names of variables (within grib message, assuming they correspond to u-wind and v-wind)
        :param mask: name of mask variable (within grib message)
        """
        self.name = npz_path.replace('.npz', '').split('_')[2]

        file = np.load(npz_path, allow_pickle=True)

        self.mask = file['mask']
        self.gribs = file['gribs']
        self.latitudes = file['latitudes']
        self.longitudes = file['longitudes']
        self.time_refs = file['time_refs']

        self.granularity = 3

    def print_mask(self, save=False, show=True, color=None, figure_name='map_mask', ax=None):
        """
        Plot mask (original 1st read layer)
        """
        if ax is None:
            ax = plt(figure_name, figsize=special_parameters.plt_default_size).gca()

        reverse_mask = (self.mask == 0).astype(np.int)

        if color is not None:
            reverse_mask[color[0], color[1]] = -15

        reverse_mask = np.ma.masked_where(reverse_mask == 1, reverse_mask)

        # Plot
        ax.imshow(np.flip(reverse_mask, axis=0), cmap=matplotlib.pyplot.get_cmap('gray'), vmin=-15, vmax=5)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Land-sea mask')

    def print_wind(self, idx, save=False, show=True, figure_name='map_wind', ax=None):
        """
        Plot the u_v_wind components as quiver (basemap) from the 1st u, v components read
        """
        if ax is None:
            ax = plt(figure_name, figsize=special_parameters.plt_default_size).gca()

        self.print_mask(show=False, save=False, ax=ax)

        x_grid = np.arange(0, self.mask.shape[1])
        y_grid = np.arange(0, self.mask.shape[0])
        x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)

        intensity = np.sqrt(np.power(np.flip(self.gribs[idx, 0, :, :], axis=0), 2) +
                            np.power(np.flip(self.gribs[idx, 1, :, :], axis=0), 2))

        u10 = np.flip(self.gribs[idx, 0, :, :], axis=0).flatten()
        v10 = np.flip(self.gribs[idx, 1, :, :], axis=0).flatten()

        wind_speed = np.sqrt((u10 ** 2 + v10 ** 2))

        # Create colour bar
        norm = matplotlib.colors.Normalize()
        norm.autoscale(wind_speed)
        cm = matplotlib.cm.CMRmap  # selecting the colourmap

        sm = matplotlib.cm.ScalarMappable(cmap=cm, norm=norm)
        sm.set_array([])

        ax.pcolormesh(x_mesh, y_mesh, intensity, alpha=0.5, cmap=cm)

        # Plot
        Y, X = np.mgrid[0:self.gribs[idx, 0].shape[0]:self.granularity, 0:self.gribs[idx, 0].shape[1]:self.granularity]
        q = ax.barbs(X, Y, np.flip(self.gribs[idx, 0, ::self.granularity, ::self.granularity], axis=0),
                      np.flip(self.gribs[idx, 1, ::self.granularity, ::self.granularity], axis=0),
                      length=3, linewidths=0.5)

        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

        cbar = ax.figure.colorbar(sm)
        cbar.set_label('velocity (m.s-1)')

        ax.set_title('Wind velocity  ' + str(self.time_refs[idx, 3]))

    def out_of_envelope(self, lon, lat, ):
        """
        :param lon: longitude of the point
        :param lat: latitude of the point
        :return: whether or not the requested is Outside the envelope of the current grib
        """
        if lon > self.longitudes.max() or lon < self.longitudes.min() or lat > self.latitudes.max() or \
                lat < self.latitudes.min():
            return True
        else:
            return False

    def get_uv_wind(self, lon, lat, idx):
        """
        Consult https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.interpolate.interpn.html

        :param lon: longitude of the point
        :param lat: latitude of the point
        :param idx: idx in the grib file (see physical_time_to_idx)
        :return: -- interpolated -- (u, v) wind conditions at (lon, lat) point, for the corresponding 'idx'
        """
        # find out idx relatively to physical time giv
        idx = idx
        u10 = self.gribs[idx, 0, :, :]
        v10 = self.gribs[idx, 1, :, :]

        interpolated_value_uwind = interpolate.interpn(points=(self.latitudes[:, 0], self.longitudes[0, :]), values=u10,
                                                       xi=([lat], [lon]), method="linear")
        interpolated_value_vwind = interpolate.interpn(points=(self.latitudes[:, 0], self.longitudes[0, :]), values=v10,
                                                       xi=([lat], [lon]), method="linear")

        return interpolated_value_uwind, interpolated_value_vwind

    def physical_time_to_idx(self, physical_timestamp):
        """
        :param physical_timestamp:
        :return: if requested time exists in this grib, return the index of the layer corresponding to it.
        """

        if physical_timestamp < min(self.time_refs[:, 3]) or physical_timestamp > max(self.time_refs[:, 3]):
            raise ValueError("Out of --time-- scope. The requested physical time is not included in this grib file.")

        i = bisect.bisect_left(self.time_refs[:, 3], physical_timestamp)
        optim_timestamp = min(self.time_refs[:, 3][max(0, i - 1): i + 2], key=lambda t: abs(physical_timestamp - t))
        return np.squeeze(np.where(self.time_refs[:, 3] == optim_timestamp))

    def in_land(self, lat, lon):
        """
        Mask is interpolated as far as the resolution might be coarser than the resolution of the game
        :param lat: latitude of point requested
        :param lon: longitude of point requested
        :return: True if point is in land, else False.
        """
        return self.mask[project(lat, lon, self)] == 1


if __name__ == "__main__":
    grib = NumpyGrib('./numpy_grib_201901160.npz')

    grib.print_wind(idx=0, save=True)
