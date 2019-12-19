import os

import numpy as np
import torch
import random

from datascience.data.datasets.util.numpy_grib import project, NumpyGrib
from datascience.data.util.source_management import check_source
from datascience.ml.neural.reinforcement.game.util.abstract_game import AbstractGame
from datascience.data.datasets.util.polar_to_numpy import Polar
from datascience.visu.util import plt, save_fig
import pandas as pd
import bisect


# very first implementation (no lag, no inception ; idea=instead of lag use forecast ;
# pressure fields could be useful aswell)
from engine.logging import print_errors
from engine.util.console.print_colors import color


class OffshoreRegatta(AbstractGame):

    def __init__(self, source, nb_try_max=1000, islands_sup=0, close_target=False, auto_restart=True):
        """
        :param root_dir: the root dir of the grib files
        :param polar: the polar path to the file
        :param nb_try_max: the number of allowed tries
        """
        super().__init__()

        self.autorestart = auto_restart

        r = check_source(source)
        if 'path' not in r:
            print_errors('The source ' + source + ' does not contain path', do_exit=True)
        if 'polar' not in r:
            print_errors('The source ' + source + ' does not contain polar', do_exit=True)

        self.root_dir = r['path']

        self.game = None

        self.numpy_grib = None
        self.polar = Polar(path_polar_file=r['polar'])

        self.target = None
        self.position = None
        self.start_position = None

        self.grib_list = [file for file in os.listdir(self.root_dir) if file.endswith('.npz')]

        self.start_timestamp = None
        self.timedelta = None
        self.track = None

        self.score = 0
        self.score_ = 0

        self.nb_try = 0
        self.nb_try_max = nb_try_max

        self.dist = None
        self.old_dist = None
        self.dir = None
        self.sog = None
        self.cog = None

        self.twa = None
        self.tws = None

        self.twd = None

        self.close_target = close_target

        self.islands_sup = islands_sup

        self.bins = np.array([i * 45 for i in range(8)])
        self.start()

    def start(self):
        # loading a random GRIB file
        self.numpy_grib = NumpyGrib(os.path.join(self.root_dir, random.choice(self.grib_list)))

        # set as the first timestamp of the grib file if not information is given
        self.start_timestamp = self.numpy_grib.time_refs[0, 3]
        self.position, self.target = self.rand_positions(self.numpy_grib, close_target=self.close_target)

        # randomize fake islands
        for i in range(self.islands_sup):
            lat = np.random.randint(0, 5)
            lon = np.random.randint(0, 5)
            center = [np.random.randint(5, self.numpy_grib.latitudes.__len__() - 5),
                      np.random.randint(5, self.numpy_grib.longitudes.__len__() - 5)]
            self.numpy_grib.mask[center[0]-lat:center[0]+lat, center[1]-lon:center[1]+lon] = 1

        while self.numpy_grib.in_land(lon=self.target[0], lat=self.target[1]) or \
                self.numpy_grib.in_land(lon=self.position[0], lat=self.position[1]):
            # regenerate init and target since at least one of them is in land
            self.position, self.target = self.rand_positions(numpy_grib=self.numpy_grib, close_target=self.close_target)

        self.start_position = np.copy(self.position)

        # initialisation of variables
        self.timedelta = pd.Timedelta(minutes=0)
        self.score = 0
        self.nb_try = 0
        self.sog = 0
        self.cog = 0
        self.twa = [0]
        self.tws = [0]
        self.twd = [0]

        # keep track for plotting
        self.track = np.expand_dims(np.array(self.position), axis=0)

        # compute first distance
        self.dist = self.distance()
        self.old_dist = self.dist
        # compute first direction
        _, self.dir = self.bearing()

    def get_view(self):
        """
        Extract a squared frame (11x11x3) of the map centered on 'self.position'.
        The third dimension relates to identification of the layer (uwind, vwind, mask, ...).

        :return: a 3D array of (11x11x3) of the map centered on 'self.position' for the corresponding physical time.
        """
        lon_idx, lat_idx = self.map_xy(self.position)  # find out the local envelope
        t_idx = self.numpy_grib.physical_time_to_idx(self.start_timestamp + self.timedelta)  # find time idx

        # view
        view = self.numpy_grib.gribs[t_idx, :, lat_idx - 5:lat_idx + 5, lon_idx - 5:lon_idx + 5]
        if view.shape[1] != 10 or view.shape[2] != 10:
            raise ValueError('View not of the correct size!')

        # stack mask to the view as well
        mask = np.expand_dims(self.numpy_grib.mask[lat_idx - 5:lat_idx + 5, lon_idx - 5:lon_idx + 5], axis=0)
        view = np.concatenate((view, mask), axis=0)

        return view  # all variables at t_idx

    def get_state(self):
        heading = np.array([self.target[0] - self.position[0], self.target[1] - self.position[1]])
        # TODO should use bearing instead ?!
        return torch.from_numpy(self.get_view()).float(), torch.from_numpy(heading).float()

    def score(self):
        # TODO rename it in reward or use only compute_reward (renaming abstract class?)
        pass

    def action(self, action, timedelta=None, auto_restart=True):
        """
        :param timedelta:
        :param auto_restart:
        :param action: the action number corresponding to a true COG (stored in self.bins)
        :return:
        """

        if timedelta is not None:
            self.timedelta = timedelta

        dead = False
        self.nb_try += 1

        # find t_index
        t_idx = self.numpy_grib.physical_time_to_idx(self.start_timestamp + self.timedelta)

        # get wind speed (tws) and direction (twd) at the position of the boat
        u10, v10 = self.numpy_grib.get_uv_wind(lon=self.position[0], lat=self.position[1], idx=t_idx)

        self.tws = np.sqrt(u10 ** 2 + v10 ** 2)
        twd_rad = np.arctan(u10/v10)

        self.twd = np.abs(twd_rad * 180 / np.pi)
        if u10 > 0 and v10 > 0:
            self.twd = 180 + self.twd
        elif u10 > 0 and v10 < 0:
            self.twd = 90 + self.twd
        elif u10 < 0 and v10 > 0:
            self.twd = self.twd + 270

        # get direction of the boat 'cog' (from action) and compute TWA at the position of the boat
        cog = self.bins[action]

        twa = self.twd - cog
        if twa < 0:
            twa += 360
        self.twa = twa

        # get approximate speed of the boat ('sog'): we assume same regime locally -- dir and intensity --
        # for the period of the action (here 10min)
        sog = self.polar.speed_boat(twa=self.twa, wind_speed=self.tws)

        self.track = np.concatenate((self.track, np.expand_dims(self.position, axis=0)), axis=0)

        # move the boat along the provided direction during a period (10min)
        self.position = self.move_boat(sog=sog, interval_sec=10 * 60, cog=cog)

        # update state
        self.old_dist = self.dist
        self.dist = self.distance()
        _, self.dir = self.bearing()
        self.sog = sog
        self.cog = cog
        self.timedelta += pd.Timedelta(minutes=10)

        # test if we are in the land
        if self.numpy_grib.in_land(lon=self.position[0], lat=self.position[1]):
            dead = True

        # test if we found the target (an area close by, round to 1000m)
        if self.dist <= 1000:
            self.score_ = self.score
            if auto_restart:
                self.start()
            print(color.GREEN + 'he won!' + color.END)
            return self.get_state(), 10, True

        if dead:
            # handle the case boat is dead.
            print(color.RED + 'he is dead' + color.END)
            self.score_ = self.score
            if auto_restart:
                self.start()  # re-initialisation
            return self.get_state(), -10, True
        elif self.nb_try >= self.nb_try_max:
            print(color.RED + 'restarting...(reached nb of tries)' + color.END)
            self.score_ = self.score
            if auto_restart:
                self.start()
            return self.get_state(), -1, True
        else:
            try:
                rew = self.compute_reward()
                self.score += rew
                return self.get_state(), rew, False
            except ValueError as e:
                self.score_ = self.score
                if auto_restart:
                    self.start()
                print(color.RED + 'restarting...(' + str(e) + '!)' + color.END)
                return self.get_state(), -1, True

    def bearing(self):
        """
        Bearing = Direction to follow to reach the target from the local position (also known as forward azimuth).
        Consult http://www.movable-type.co.uk/scripts/latlong.html for details.
        :return: both the bin (action) corresponding to the true bearing AND the true bearing
        """
        # http://www.movable-type.co.uk/scripts/latlong.html
        to_rad = np.pi / 180
        rad_position_lat = self.position[1] * to_rad
        rad_position_lon = self.position[0] * to_rad
        rad_target_lat = self.target[1] * to_rad
        rad_target_lon = self.target[0] * to_rad

        delta = rad_target_lon - rad_position_lon

        theta = np.arctan2(
            np.sin(delta) * np.cos(rad_target_lat),
            (
                    np.cos(rad_position_lat) * np.sin(rad_target_lat) -
                    np.sin(rad_position_lat) * np.cos(rad_target_lat) * np.cos(delta)
            )
        )
        b = theta * 1 / to_rad

        b = (360 + b) % 360
        boat_dir_bin = np.digitize(b, self.bins, True) - 1
        return boat_dir_bin, b

    def distance(self, node=None):
        """
        Provide the distance (nautical miles) between the current position of the boat and the target.
        :return: a distance (nautical miles) between the current position and the target
        """

        if node is not None:
            position = node
        else:
            position = self.position

        to_rad = np.pi / 180
        rad_position_lat = position[1] * to_rad
        rad_position_lon = position[0] * to_rad
        rad_target_lat = self.target[1] * to_rad
        rad_target_lon = self.target[0] * to_rad

        r = 6371000  # mean earth radius (appproximated as 6371km)
        dist = r * np.arccos(
            np.sin(rad_position_lat) * np.sin(rad_target_lat) +
            np.cos(rad_position_lat) * np.cos(rad_target_lat) * np.cos(rad_position_lon-rad_target_lon)
        )
        return dist/1852.

    def move_boat(self, sog, interval_sec, cog):
        """
        Move the boat from its position to the new position accordingly to the given parameters:
        The boat will move for a period of 'interval_sec', TOWARDS 'cog' and at a speed of 'sog'.
        :param sog: speed over ground
        :param interval_sec: interval of time (in second)
        :param cog: course(direction) over ground
        :return: the new position of the boat
        """
        r = 6371000  # mean earth radius (approximated as 6371km)
        speed_si = sog * 1852/3600  # knots to m.s^-1
        d = speed_si * interval_sec
        delta = d / r  # angular distance d/r where d is the distance being travelled
        cog = cog if cog < 180 else cog - 360
        to_rad = np.pi / 180
        to_deg = 180 / np.pi
        rad_position_lat = self.position[1] * to_rad
        rad_position_lon = self.position[0] * to_rad
        cog_rad = cog * to_rad
        # new latitude
        lat = np.arcsin(
            np.sin(rad_position_lat) * np.cos(delta) + np.cos(rad_position_lat) * np.sin(delta) * np.cos(cog_rad)
        )

        # new longitude
        lon = rad_position_lon + np.arctan2(
            np.sin(cog_rad) * np.sin(delta) * np.cos(rad_position_lat),
            np.cos(delta) - np.sin(rad_position_lat) * np.sin(lat)
        )

        lat *= to_deg
        lon *= to_deg

        new_position = [lon, lat]

        return new_position

    def map_xy(self, position):
        """
        :param position: (lon,lat) tuple, within the numpy_grib envelope
        :return: x,y index of the position in 'self.numpy_grib'
        """
        lons = self.numpy_grib.longitudes[0, :]
        lats = self.numpy_grib.latitudes[:, 0]

        # lon
        i = bisect.bisect_left(lons, position[0])
        optim_lon = min(lons[max(0, i - 1): i + 2], key=lambda t: abs(position[0] - t))
        lon_idx = np.squeeze(np.where(lons == optim_lon))

        # lat
        j = bisect.bisect_left(lats, position[1])
        optim_lat = min(lats[max(0, j - 1): j + 2], key=lambda t: abs(position[1] - t))
        lat_idx = np.squeeze(np.where(lats == optim_lat))

        return lon_idx, lat_idx

    @staticmethod
    def rand_positions(numpy_grib, close_target=False):
        """
        Notice, locations are given as tuple of (lon, lat).
        We also force the positions to not be too much on the edges (because we had to deal with patches of the same
        size (see get_view()).

        # TODO here we assume the resolution of the computational-mesh is of 0.5 degrees. Need to adapt to all res.

        :param numpy_grib: forecast on which game is played.
        :param close_target: set if target has to be close from the start point
        :return: two random locations within the geographical envelope of 'numpy_grib'
        """
        start = np.array([round(np.random.uniform(numpy_grib.longitudes.min() + 5, numpy_grib.longitudes.max() - 5), ndigits=6),
                 round(np.random.uniform(numpy_grib.latitudes.min() + 5, numpy_grib.latitudes.max() - 5), ndigits=6)])

        target = np.array([round(np.random.uniform(numpy_grib.longitudes.min() + 5, numpy_grib.longitudes.max() - 5), ndigits=6),
                  round(np.random.uniform(numpy_grib.latitudes.min() + 5, numpy_grib.latitudes.max() - 5), ndigits=6)])

        if close_target:
            target = (start + np.random.uniform(-3, +3, size=(2,))).tolist()

        return start, target

    def compute_reward(self):
        # TODO Est ce qu'il ne faudrait pas standardiser par rapport à la distance initiale ?
        return self.old_dist - self.dist

    def __str__(self):
        result = color.BLUE + "Date: " + color.END + str(self.start_timestamp + self.timedelta) + '\n'
        result += color.BLUE + "Position (lon, lat): " + color.END + str(self.position) + '\n'
        result += color.BLUE + "Target (lon,lat): " + color.END + str(self.target) + '\n'
        result += color.BLUE + "Distance to target: " + color.END + str(round(self.dist)) + "m" + '\n'
        result += color.BLUE + "Bearing to target: " + color.END + str(round(self.dir)) + "°" + '\n'
        result += color.BLUE + "SOG, COG: " + color.END + str(round(self.sog)) + "," + str(round(self.cog)) + "°\n"
        result += color.BLUE + "TWA, TWS, TWD: " + color.END + str(round(self.twa[0])) + "°,"
        result += str(round(self.tws[0])) + "," + str(round(self.twd[0])) + '°'
        return result

    def print(self):
        print(self)

    def plot(self, plot_weather=False, save=False, show=True, track=None, figure_name='regatta_plot'):
        if track is not None:
            self.track = track

        # draw land
        if plot_weather:
            # current simulation timestamp
            t_idx = self.numpy_grib.physical_time_to_idx(self.start_timestamp + self.timedelta)
            self.numpy_grib.print_wind(show=False, save=False, idx=t_idx)
        else:
            self.numpy_grib.print_mask(show=False, save=False)

        # current position
        y_position, x_position = project(latitude=self.position[1], longitude=self.position[0], grib=self.numpy_grib)

        y_position = self.numpy_grib.latitudes.shape[0] - y_position

        fig = plt(figure_name)

        fig.plot(x_position, y_position, 's', markersize=4, color="blue")

        # target
        y_target, x_target = project(latitude=self.target[1], longitude=self.target[0], grib=self.numpy_grib)

        y_target = self.numpy_grib.latitudes.shape[0] - y_target
        fig.plot(x_target, y_target, 'bo', markersize=6, color="green")

        # track
        X, Y = [], []
        for lat, lon in zip(self.track[:, 1], self.track[:, 0]):
            y, x = project(lat, lon, self.numpy_grib)
            X.append(x)
            Y.append(self.numpy_grib.latitudes.shape[0] - y)

        fig.plot(X, Y, '-', color="red")
        fig.xlabel('Longitude')
        fig.ylabel('Latitude')

        fig.suptitle('Offshore Regatta (at ' + str(self.start_timestamp + self.timedelta) + ')', y=1, fontsize=12)

        save_fig(figure_name=figure_name)

    def save_plot(self, path):
        pass

    def show_view(self, figure_name='regatta_view', show=False):
        view = self.get_view()

        uwind = view[0, :, :]
        vwind = view[1, :, :]
        mask = view[2, :, :]

        print(view.shape)

        fig = plt(figure_name)

        # ax = fig.gca()

        fig.subplot(1, 3, 1)
        fig.imshow(np.squeeze(uwind))
        fig.title('u wind (m.s-1)')

        fig.subplot(1, 3, 2)
        fig.imshow(np.squeeze(vwind))
        fig.title('v wind (m.s-1)')

        fig.subplot(1, 3, 3)
        fig.imshow(np.squeeze(mask))
        fig.title('mask')

        save_fig(figure_name=figure_name)


# playing test
if __name__ == '__main__':
    # create polar

    polar_path = os.path.join(os.path.dirname(__file__), "../rawdata/polar", "class40.pol")
    regatta = OffshoreRegatta(root_dir='../rawdata/', polar=polar_path)

    regatta.print()

    # play
    n_step = 700
    for _ in range(n_step):
        act, _ = regatta.bearing()
        regatta.action(act)
        if _ % 100 == 0:
            print(_)
            regatta.print()

    regatta.show_view()
    regatta.plot(plot_weather=True, save=True)
    print(regatta)
