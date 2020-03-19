import os
import cv2
import math
import datetime
import time as ti
import numpy as np
import matplotlib.pyplot as plt

EQUATOR_ARC_SECOND_IN_METERS = 30.87  # meters
TILES_SIZE = 3600  # px
RUNNING_TILES_COEF = 10  # considering smaller virtual tiles of 0.1° resolution (to be able resizing)


class ExtractionError(Exception):
    """
    Exception raised when any problem occur during the extraction of a patch
    """
    def __init__(self, lat, lng, id=None, message='Error while extracting a patch'):
        super().__init__(message)
        self.message = message
        self.lat = lat,
        self.lng = lng
        if id is not None:
            self.id = id
        else:
            self.id = "NA"

    def __str__(self):
        return "{} (lat:{}, lng:{}, id:{})".format(self.message, self.lat, self.lng, self.id)


class Tile(object):
    def __init__(self, tile_path):
        self.path = tile_path
        self.decomposed_path = tile_path.split("/")
        self.name = self.decomposed_path[-1]  # name should be like 'N43E003.hgt'

        # use name to define bottom left corner lat and lng (min lat and min lng of the tile)
        decomposed_name = [self.name[0], self.name[1:3], self.name[3], self.name[4:7]]
        self.lat = int(decomposed_name[1]) if decomposed_name[0] == "N" else -int(decomposed_name[1])
        self.lng = int(decomposed_name[3]) if decomposed_name[2] == "E" else -int(decomposed_name[3])
        self.loaded = False
        self.data = None
        self.running_tiles = []

    def load(self):
        size = os.path.getsize(self.path)
        dim = int(math.sqrt(size / 2))

        # load and remove the overlap of 1px
        self.data = np.fromfile(self.path, np.dtype('>i2'), dim * dim).reshape((dim, dim))[:-1, :-1]
        self.loaded = True
        return self

    def unload(self):
        self.data = None
        self.loaded = False
        for running_tile in self.running_tiles:
            running_tile.unload()

    def split(self, coef):
        self.map = np.empty((coef, coef), dtype=object)
        split_size = int(TILES_SIZE/coef)
        splits = [s for s in range(0, TILES_SIZE, split_size)]
        for i in range(len(splits)):
            for j in range(len(splits)):
                pos = (i, j)
                self.map[i, j] = RunningTile(self, pos, split_size)
        return self.map

    def show(self):
        plt.imshow(self.data)
        plt.show()

    def __str__(self):
        return 'Tile(lat-{}-{}_lng-{}-{})'.format(self.lat, self.lat+1, self.lng, self.lng+1)

    def __repr__(self):
        return self.__str__()


class RunningTile(object):
    def __init__(self, tile, pos, size):
        self.tile = tile
        self.tile.running_tiles.append(self)
        self.pos = pos
        self.size = size
        self.range = 1.0/RUNNING_TILES_COEF
        self.lat = tile.lat - (pos[0] * (1.0/RUNNING_TILES_COEF)) + 1 - (1.0/RUNNING_TILES_COEF)
        self.lng = tile.lng + (pos[1] * (1.0/RUNNING_TILES_COEF))
        self.loaded = False
        self.data = None

    def load(self):
        if self.tile.loaded:
            self.data = self.tile.data[self.pos[0]*self.size:self.pos[0]*self.size+self.size, self.pos[1]*self.size:self.pos[1]*self.size+self.size]
        else:
            self.data = self.tile.load().data[self.pos[0]*self.size:self.pos[0]*self.size+self.size, self.pos[1]*self.size:self.pos[1]*self.size+self.size]
        self.loaded = True
        return self

    def unload(self):
        self.loaded = False
        self.data = None

    def show(self):
        plt.imshow(self.data)
        plt.show()

    def __str__(self):
        return 'RunningTile(lat-{:.1f}-{:.1f}_lng-{:.1f}-{:.1f})'.format(self.lat, self.lat+(1.0/RUNNING_TILES_COEF),
                                                                         self.lng, self.lng+(1.0/RUNNING_TILES_COEF))

    def __repr__(self):
        return self.__str__()


class TileManager(object):
    def __init__(self, tiles_dir_path, max_cache=100):
        self.tiles_dir_path = tiles_dir_path
        self.max_cache = max_cache
        self.cache_list = []
        self.cache_dict = {}

        list_dir_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.tiles_dir_path)) for f in fn]

        # get list of tiles and coordinates bounds
        self.list_tiles = []
        self.min_lat = 100
        self.max_lat = -100
        self.min_lng = 200
        self.max_lng = -200
        for i, item in enumerate(list_dir_files):
            if item.endswith(".hgt"):
                tile = Tile(item)
                if tile.lat < self.min_lat:
                    self.min_lat = tile.lat
                elif tile.lat > self.max_lat:
                    self.max_lat = tile.lat
                if tile.lng < self.min_lng:
                    self.min_lng = tile.lng
                elif tile.lng > self.max_lng:
                    self.max_lng = tile.lng
                self.list_tiles.append(tile)
        self.max_lat += 1  # add 1 to max for the last tile dimension
        self.max_lng += 1
        self.size_lat = self.max_lat - self.min_lat
        self.size_lng = self.max_lng - self.min_lng

        # create spatial matrix to order and place tiles relatively to their spatial position
        self.map = np.empty((self.size_lat, self.size_lng), dtype=object)

        self.running_map = np.empty((self.size_lat*RUNNING_TILES_COEF, self.size_lng*RUNNING_TILES_COEF), dtype=object)
        for tile in self.list_tiles:
            pos_lat = self.size_lat - (tile.lat - self.min_lat) - 1  # reverse lat (increase from bottom to top)
            pos_lng = tile.lng - self.min_lng
            self.running_map[pos_lat*RUNNING_TILES_COEF:(pos_lat+1)*RUNNING_TILES_COEF,
                             pos_lng*RUNNING_TILES_COEF:(pos_lng+1)*RUNNING_TILES_COEF] = tile.split(RUNNING_TILES_COEF)
        self.running_size_lat = self.size_lat * RUNNING_TILES_COEF
        self.running_size_lng = self.size_lng * RUNNING_TILES_COEF

    def read_tile(self, pos, corrected_data_running_shape):
        tile = self.running_map[pos[0], pos[1]]
        if not tile.loaded:
            patch = tile.load().data
        else:
            patch = tile.data

        # TODO: find better function to resize
        patch = cv2.resize(np.stack((patch, patch, patch), axis=2), dsize=(corrected_data_running_shape[1], corrected_data_running_shape[0]),
                           interpolation=cv2.INTER_LINEAR)
        patch = patch[:, :, 2]
        return patch

    def extract(self, lat, lng, size, resolution):
        # conversion arc/second to meters
        data_pixel_size_lat = EQUATOR_ARC_SECOND_IN_METERS  # constant for lat
        data_pixel_size_lng = EQUATOR_ARC_SECOND_IN_METERS * math.cos(math.radians(lat))  # depends on lat for lng
        # for the extraction of one patch, all data are considered from this resolution

        running_tiles_size = int(TILES_SIZE / RUNNING_TILES_COEF)
        corrected_data_running_shape = (round(running_tiles_size * (data_pixel_size_lat / resolution)),
                                        round(running_tiles_size * (data_pixel_size_lng / resolution)))

        tile_pos_lat = self.running_size_lat - int((lat - self.min_lat) * RUNNING_TILES_COEF) - 1
        tile_pos_lng = int((lng - self.min_lng) * RUNNING_TILES_COEF)

        center_tile = self.running_map[tile_pos_lat, tile_pos_lng]
        pixel_lat = round(corrected_data_running_shape[0] - ((lat-center_tile.lat) * corrected_data_running_shape[0] / center_tile.range))
        pixel_lng = round(
            ((lng - center_tile.lng) * corrected_data_running_shape[1] / center_tile.range))

        # TODO: take into account odd number size
        min_lat = int(pixel_lat - int(size/2))
        min_lng = int(pixel_lng - int(size/2))
        max_lat = int(pixel_lat + int(size/2))
        max_lng = int(pixel_lng + int(size/2))

        center_image_pos_x, center_image_pos_y = 0, 0
        aggregation_size_x, aggregation_size_y = 1, 1
        while min_lat < 0 or max_lat >= corrected_data_running_shape[0] * aggregation_size_x or \
                min_lng < 0 or max_lng >= corrected_data_running_shape[1] * aggregation_size_y:
            if min_lat < 0:
                max_lat = max_lat + corrected_data_running_shape[0]
                min_lat = min_lat + corrected_data_running_shape[0]
                aggregation_size_x += 1
                center_image_pos_x += 1
            if min_lng < 0:
                max_lng = max_lng + corrected_data_running_shape[1]
                min_lng = min_lng + corrected_data_running_shape[1]
                aggregation_size_y += 1
                center_image_pos_y += 1
            if max_lat >= corrected_data_running_shape[0] * aggregation_size_x:
                aggregation_size_x += 1
            if max_lng >= corrected_data_running_shape[1] * aggregation_size_y:
                aggregation_size_y += 1

        list_im = []
        for i in range(aggregation_size_x):
            for j in range(aggregation_size_y):
                relative_x = i - center_image_pos_x
                relative_y = j - center_image_pos_y
                """
                if self.is_not_tile(x + relative_x, y + relative_y):
                    print("error here")
                    raise ExtractionError(x_lamb, y_lamb, error_type=1, identifier=identifier)
                """
                list_im.append(((tile_pos_lat + relative_x, tile_pos_lng + relative_y),
                                (corrected_data_running_shape[0] * i, corrected_data_running_shape[0] * (i + 1),
                                 corrected_data_running_shape[1] * j, corrected_data_running_shape[1] * (j + 1))))

        aggregation_im = np.ndarray((corrected_data_running_shape[0] * aggregation_size_x,
                                     corrected_data_running_shape[1] * aggregation_size_y),
                                    dtype="int16")
        for im_tile in list_im:
            aggregation_im[im_tile[1][0]:im_tile[1][1], im_tile[1][2]:im_tile[1][3]]\
                = self.read_tile((im_tile[0][0], im_tile[0][1]), corrected_data_running_shape)
        patch = aggregation_im[min_lat:max_lat, min_lng:max_lng]

        return patch

    def extract_patches(self, long_lat_df, destination_directory, res, size=256, check_file=True):
        """
        The main extraction method for multiple extractions
        :param long_lat_df:
        :param destination_directory:
        :param size:
        :param step:
        :param error_extract_folder:
        :param error_cache_size:
        :param white_percent_allowed:
        :param check_file:
        """

        err_manager = _ErrorManager(destination_directory)

        if not os.path.exists(destination_directory):
            os.makedirs(destination_directory)

        total = long_lat_df.shape[0]
        start = datetime.datetime.now()
        extract_time = 0

        for idx, row in enumerate(long_lat_df.iterrows()):
            longitude, latitude, occ_id = row[1][0], row[1][1], row[1][2]

            if idx % 10000 == 9999:
                _print_details(idx+1, total, start, extract_time, latitude, longitude)

            patch_id = int(row[1][2])

            # constructing path with hierarchical structure
            path = _write_directory(destination_directory, str(patch_id)[-2:], str(patch_id)[-4:-2])

            patch_path = os.path.join(path, str(patch_id) + ".npy")

            # if file exists pursue extraction
            if os.path.isfile(patch_path) and check_file:
                continue

            t1 = ti.time()
            t2 = 0
            try:
                patch = self.extract(latitude, longitude, size, res)
            except Exception as err:
                t2 = ti.time()
                err_manager.append(latitude, longitude, occ_id)
            else:
                t2 = ti.time()
                np.save(patch_path, patch)
            finally:
                delta = t2 - t1
                extract_time += delta

            err_manager.write_errors()
        print("end")
        _print_details(total, total, start, extract_time, 0.0, 0.0)


class _ErrorManager(object):
    def __init__(self, path, cache_size=1000):
        # setting up the destination file
        current_datetime = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        error_extract_file = path + current_datetime + '_errors.csv'
        with open(error_extract_file, "w") as file:
            file.write(str(current_datetime) + "Occ_id;Latitude;Longitude\n")
        self.file = error_extract_file

        # setting up the error cache
        self.error_cache = []
        self.total_size = 0

        # the maximum number of elements in the cache
        self.cache_size = cache_size

    def __len__(self):
        return self.total_size

    def append(self, lat, lng, id):
        """
        :param error:
        :return:
        """

        self.error_cache.append('{};{};{}'.format(id, lat, lng))
        self.total_size += 1
        if len(self.error_cache) >= self.cache_size:
            self.write_errors()

    def write_errors(self):
        with open(self.file, "a") as file:
            for err in self.error_cache:
                file.write(err + "\n")
        self.error_cache = []


def _write_directory(root, *args):
    path = root
    for d in args:
        path = os.path.join(path, d)
        if not os.path.exists(path):
            os.makedirs(path)
    return path


def _print_details(idx, total, start, extract_time, latitude, longitude):
    time = datetime.datetime.now()
    print('\n{}/{}'.format(idx, total))
    p = ((idx - 1) / total) * 100
    print('%.2f' % p)
    delta = (time - start).total_seconds()
    estimation = (delta * total) / idx
    date_estimation = start + datetime.timedelta(seconds=estimation)
    print('mean extraction time: {}'.format(extract_time / idx))
    print('Actual position: {}'.format((latitude, longitude)))
    print('Time: {}, ETA: {}'.format(datetime.timedelta(seconds=delta), date_estimation))


if __name__ == "__main__":
    dir_name = '/home/bdeneu/alti/'
    t_manager = TileManager(dir_name)
    p = t_manager.extract(45.546314239502, 6.79009008407593, 256, 1.0)
    np.save("10041069744.npy", p)
    #plt.imshow(p)
    #plt.show()


    """
    file = '/home/bdeneu/data/alti/N43E003.hgt'
    file2 = '/home/bdeneu/data/alti/N43E004.hgt'

    tile = Tile(file)
    data = tile.load()

    tile2 = Tile(file2)
    data_2 = tile2.load()

    print(data)
    print(data.shape)

    print(data_2)
    print(data_2.shape)

    plt.imshow(np.log(data+10))
    plt.show()

    lon_size = 30.87 * math.cos(math.radians(43.6))
    print(lon_size)

    data_size = (data.shape[0] * 30.87, data.shape[1] * lon_size)
    print(data_size)


    # print(data)

    #data2 = cv2.resize(np.stack((data, data, data), axis=2), dsize=(6000, 6000), interpolation=cv2.INTER_LINEAR)
    data2 = cv2.resize(np.stack((data, data, data), axis=2), dsize=(int(data_size[1]/8), int(data_size[0]/8)), interpolation=cv2.INTER_LINEAR)

    # print(data2)
    data2 = data2[:, :, 2]

    plt.imshow(np.log(data2+10))
    plt.show()
    """
