import math
import PIL.Image
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import datetime
from scipy.sparse import bsr_matrix, lil_matrix, save_npz

from engine.path import list_files, output_path
from engine.logging.logs import print_info, print_info, print_errors
from pyproj import Transformer, Proj
import time as ti


class ExtractionError(Exception):
    """
    Exception raised when any problem occur during the extraction of a patch
    """
    def __init__(self, x, y, error_type=1, identifier=None, message=''):
        """
        :param identifier:
        :param error_type:
        :param x:
        :param y:
        :param error_type: 1 = partially outside of the area; 2 = center outside of the area;
                           3 = on the area but out of data coverage
        """
        super().__init__(message)
        self.x = x,
        self.y = y
        self.error_type = error_type,
        if identifier is not None:
            self.identifier = identifier
        else:
            self.identifier = "NA"


# resolutions of the IGN maps // size (m), size (px), resolution (m)
resolution_details = {
    '5M00': (10000, 2000, 5.0),
    '0M50': (5000, 10000, 0.5),
    '0M20': (5000, 25000, 0.2)
}

# image types and layers
image_type_details = {
    'RVB': ('red', 'green', 'blue'),
    'IRC': ('near_ir', 'red', 'green')
}


class Tile(object):
    """
    Contains infos about a tile/an image

    list of attributes:
    - image_name : the absolute path to the corresponding tile,
    - decomposed_name : the path in the form of a list of directories up to the file itself,
    - department : the french department visible on the tile,
    - date : the date of the view,
    - x_min/x_max : minimum and maximum latitude wise,
    - y_min/y_max : minimum and maximum longitude wise.
    """
    def __init__(self, image_name):
        self.image_name = image_name
        self.decomposed_name = image_name.split("/")
        data_value = self.decomposed_name[-1].split("-", 6)
        self.department = data_value[0]
        self.y_min = int(data_value[2]) * 1000
        self.y_max = None
        self.x_max = int(data_value[3]) * 1000
        self.x_min = None
        self.date = str("".join(self.decomposed_name[-3].split("_")[-1].split("-")[0:2]))

        info_image = self.decomposed_name[-2].split("_")
        self.image_resolution = info_image[2]
        self.image_type = info_image[1]
        self.image_encoding = info_image[3]
        self.image_projection = info_image[4]

        # size and range of images depends on the ign dataset
        global resolution_details
        self.image_range, self.image_size, self.resolution = resolution_details[self.image_resolution]

    def set_x_min(self, x_min):
        self.x_min = x_min

    def set_y_max(self, y_max):
        self.y_max = y_max

    def __str__(self):
        return 'Tile({}_{}_{}_{})'.format(self.department, self.y_min, self.x_max, self.date)

    def __repr__(self):
        return self.__str__()


class IGNImageManager(object):

    """
    Manages IGN images

    liste attibuts :
    - imdir : le chemin d'accès au dossier contenant toutes les miages de la zones détude. Les images doivent toutes
    être du même type (RVB/IR et et même résolution)
    - list_tile : la liste des tiles de toutes la zone d'étude
    - min_x/max_x : les minimum/maximum nord-sud (croissant vers le nord) couvert par la zone détude
    - min_y/max_y : les minimum/maximum est-ouest (croissant vers l'est) couvert par la zone détude
    - image_resolution : résolution des images en m/pixels
    - image_type : le type d'image, RVB ou IR
    - image_encoding : l'encodage des images
    - image_projection : la projection utilisée pour les images (normalement LA93)
    - image_range : le rang de valeurs de projection couvert par une tile
    - image_size : le nb de pixels en largeur et longeur d'une tile
    - carto : matrice contenant les tiles ordonnées par leur position relative
    """

    def __init__(self, image_dir, max_cache=1000, in_proj='epsg:4326'):
        self.image_dir = image_dir
        
        # config geographic projections
        self.in_proj, self.ign_proj = Proj(init=in_proj), Proj(init='epsg:2154')

        # self.transformer_in_out = Transformer.from_proj(in_proj, out_proj)
        self.transformer = Transformer.from_proj(self.in_proj, self.ign_proj)

        # configure cache of tiles
        self.cache_dic = {}
        self.cache_list = []

        self.max_cache = max_cache
        
        # get all files in the folder image directory
        files = list_files(self.image_dir)
        print(files)
        
        # construct tile object for all tile images in the list of files
        self.list_tile, \
            self.min_x, \
            self.max_x, \
            self.min_y, \
            self.max_y = \
            self.get_tile_list(files, get_bounds=True)

        print_info('{} tiles were found!'.format(len(self.list_tile)))
        
        # use the first tile in the list to get images information
        # the information are contained in the last folder's name
        info_images = self.list_tile[0].decomposed_name[-2].split("_")
        self.image_resolution = info_images[2]
        self.image_type = info_images[1]
        self.image_encoding = info_images[3]
        self.image_projection = info_images[4]

        # size and range of images depends on the ign dataset
        global resolution_details
        self.image_range, self.image_size, self.resolution = resolution_details[self.image_resolution]

        PIL.Image.MAX_IMAGE_PIXELS = self.image_size**2

        # configure the spatial area containing all tiles
        self.max_y = self.max_y + self.image_range
        self.min_x = self.min_x - self.image_range

        self.x_size = int((self.max_x-self.min_x)/self.image_range)
        self.y_size = int((self.max_y-self.min_y)/self.image_range)

        # create spatial matrix to order and place tiles relatively to their spatial position
        self.map = np.empty((self.x_size, self.y_size), dtype=object)

        # fill the cartographic matrix with all tiles
        for tile in self.list_tile:
            tile.set_y_max(tile.y_min+self.image_range)
            tile.set_x_min(tile.x_max-self.image_range)
            pos_x, pos_y = self.position_to_tile_index(tile.x_max, tile.y_min)
            if type(self.map[pos_x, pos_y]) is not Tile \
                    or self.map[pos_x, pos_y].resolution < tile.resolution \
                    or (int(self.map[pos_x, pos_y].resolution) <= int(tile.resolution) \
                    and int(self.map[pos_x, pos_y].date) < int(tile.date)):
                # if 2 tiles cover the same area we keep the most recent one
                self.map[pos_x, pos_y] = tile

    @staticmethod
    def get_tile_list(files_list, get_bounds):
        """
        returns the list of tiles object from a list of files.
        If some files do not correspond to tiles, they are ignored
        :param files_list:
        :param get_bounds:
        :return:
        """
        list_tile = []
        min_y = None
        max_y = None
        min_x = None
        max_x = None
        for i, item in enumerate(files_list):
            if item.endswith(".jp2") or item.endswith(".tif"):
                tile = Tile(item)
                if min_y is None or tile.y_min < min_y:
                    min_y = tile.y_min
                elif max_y is None or tile.y_min > max_y:
                    max_y = tile.y_min
                if min_x is None or tile.x_max < min_x:
                    min_x = tile.x_max
                elif max_x is None or tile.x_max > max_x:
                    max_x = tile.x_max
                list_tile.append(tile)

        if get_bounds:
            return list_tile, min_x, max_x, min_y, max_y
        else:
            return list_tile

    def position_to_tile_index(self, x_in, y_in):
        x_out = int((self.max_x - x_in) / self.image_range)
        y_out = int((y_in-self.min_y)/self.image_range)
        return x_out, y_out

    def is_not_tile(self, x, y):
        if x < 0 or x > self.x_size - 1 or y < 0 or y > self.y_size - 1 or type(self.map[x, y]) is not Tile:
            return True
        return False

    def get_image_at_location(self, x_lat, y_long):
        y2, x2 = self.transformer.transform(y_long, x_lat)
        # x2,y2 = int(x2),int(y2)
        print(x2, y2)
        x, y = self.position_to_tile_index(x2, y2)
        print(x, y)
        if x < 0 or x > self.x_size-1 or y < 0 or y > self.y_size-1 or type(self.map[x, y]) is not Tile:
            print_errors("Outside study zone...", do_exit=True)
        return self.map[x, y]

    def read_tile(self, pos, res):
        if pos in self.cache_dic:
            im_tile = self.cache_dic[pos]
        else:
            tile = self.map[pos[0], pos[1]]
            im_tile = plt.imread(self.map[pos[0], pos[1]].image_name)
            im_tile = cv2.resize(im_tile, dsize=(int(tile.image_range/res), int(tile.image_range/res)))

            if len(self.cache_list) >= self.max_cache:
                self.cache_dic.pop(self.cache_list[0])
                self.cache_list.pop(0)
                self.cache_list.append(pos)
                self.cache_dic[pos] = im_tile
            else:
                self.cache_list.append(pos)
                self.cache_dic[pos] = im_tile
        print(self.map[pos[0], pos[1]])
        return im_tile

    def extract_patch(self, latitude, longitude, res, size, identifier=-1, white_percent_allowed=20):
        y, x = self.transformer.transform(longitude, latitude)
        return self.extract_patch_lambert93(x, y, res, size, identifier, white_percent_allowed=white_percent_allowed)

    def extract_patch_lambert93(self, x_lamb, y_lamb, res, size, identifier=-1,
                                white_percent_allowed=20, verbose=False):
        x, y = self.position_to_tile_index(x_lamb, y_lamb)

        if self.is_not_tile(x, y):
            raise ExtractionError(x_lamb, y_lamb, error_type=2, identifier=identifier)

        # im = self.read_tile((x, y))

        pixel_y = int((y_lamb - self.map[x, y].y_min) / res)
        pixel_x = int((self.map[x, y].x_max - x_lamb) / res)

        modulo = size % 2
        half_size = int(size / 2)

        x_min = pixel_x - half_size
        x_max = pixel_x + half_size + modulo
        y_min = pixel_y - half_size
        y_max = pixel_y + half_size + modulo

        center_image_pos_x, center_image_pos_y = 0, 0
        aggregation_size_x, aggregation_size_y = 1, 1

        image_pixel_size = int(self.image_range * res)

        while x_min < 0 or x_max >= image_pixel_size*aggregation_size_x or\
                y_min < 0 or y_max >= image_pixel_size*aggregation_size_y:
            if x_min < 0:
                x_max = x_max + image_pixel_size
                x_min = x_min + image_pixel_size
                aggregation_size_x += 1
                center_image_pos_x += 1
            if y_min < 0:
                y_max = y_max + image_pixel_size
                y_min = y_min + image_pixel_size
                aggregation_size_y += 1
                center_image_pos_y += 1
            if x_max >= image_pixel_size*aggregation_size_x:
                aggregation_size_x += 1
            if y_max >= image_pixel_size*aggregation_size_y:
                aggregation_size_y += 1

        if verbose:
            print("number of tiles to load", aggregation_size_x*aggregation_size_y)

        list_im = []
        for i in range(aggregation_size_x):
            for j in range(aggregation_size_y):
                relative_x = i - center_image_pos_x
                relative_y = j - center_image_pos_y
                if self.is_not_tile(x + relative_x, y + relative_y):
                    print("error here")
                    raise ExtractionError(x_lamb, y_lamb, error_type=1, identifier=identifier)
                list_im.append(((x + relative_x, y + relative_y),
                               (image_pixel_size*i, image_pixel_size*(i+1), image_pixel_size*j, image_pixel_size*(j+1))))

        aggregation_im = np.ndarray((image_pixel_size*aggregation_size_x, image_pixel_size*aggregation_size_y, 3),
                                    dtype=int)

        for im_tile in list_im:
            aggregation_im[im_tile[1][0]:im_tile[1][1], im_tile[1][2]:im_tile[1][3]]\
                = self.read_tile((im_tile[0][0], im_tile[0][1]), res)

        patch = aggregation_im[x_min:x_max, y_min:y_max, :]

        if (np.sum(np.all(patch == 255, axis=2))/(patch.shape[0]*patch.shape[1]))*100 > white_percent_allowed:
            raise ExtractionError(x_lamb, y_lamb, error_type=3, identifier=identifier)
        return patch

    def extract_patches(self, long_lat_df, destination_directory, res, size=64, error_extract_folder=None,
                        error_cache_size=1000, white_percent_allowed=20, check_file=True):
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

        if not os.path.exists(destination_directory):
            os.makedirs(destination_directory)

        error_manager = _ErrorManager(self.in_proj,
                                      self.ign_proj,
                                      destination_directory if error_extract_folder is None else error_extract_folder,
                                      cache_size=error_cache_size
                                      )

        total = long_lat_df.shape[0]
        start = datetime.datetime.now()
        extract_time = 0

        for idx, row in enumerate(long_lat_df.iterrows()):
            longitude, latitude = row[1][0], row[1][1]

            if idx % 100000 == 99999:
                _print_details(idx+1, total, start, extract_time, latitude, longitude, len(error_manager))

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
                patch = self.extract_patch(latitude, longitude, res, size, identifier=int(patch_id),
                                           white_percent_allowed=white_percent_allowed)
            except ExtractionError as err:
                t2 = ti.time()
                error_manager.append(err)
            else:
                t2 = ti.time()
                np.save(patch_path, patch)
            finally:
                delta = t2 - t1
                extract_time += delta

        error_manager.write_errors()

    # TODO : code to produce tiff
    """
    def create_tiff(self):
        copy_carto = np.transpose(self.carto)
        raster = np.zeros((copy_carto.shape[0]*self.image_size, copy_carto.shape[1]*self.image_size, 2), dtype=np.uint8)
        print(raster.shape)
        for row in progressbar.progressbar(copy_carto):
            for tuile in row:
                if tuile is not None:
                    raster[tuile.pos_y:tuile.pos_y+self.image_size, tuile.pos_x:tuile.pos_x+self.image_size]\
                        = self.readImage((tuile.pos_x, tuile.pos_y))[:, :, :1]
    """

    def create_sparse(self, long_lat_df, size=64, step=1, error_extract_folder=None,
                        error_cache_size=1000, white_percent_allowed=20, check_file=True):
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

        error_manager = _ErrorManager(self.in_proj,
                                      self.ign_proj,
                                      output_path('errors/') if error_extract_folder is None else error_extract_folder,
                                      cache_size=error_cache_size
                                      )

        """
        max_lat_loc = long_lat_df.loc[[long_lat_df['Latitude'].idxmax()]]
        max_lat_long, max_lat_lat = float(max_lat_loc['Longitude']), float(max_lat_loc['Latitude'])

        min_lat_loc = long_lat_df.loc[[long_lat_df['Latitude'].idxmin()]]
        min_lat_long, min_lat_lat = float(min_lat_loc['Longitude']), float(min_lat_loc['Latitude'])

        max_long_loc = long_lat_df.loc[[long_lat_df['Longitude'].idxmax()]]
        max_long_long, max_long_lat = float(max_long_loc['Longitude']), float(max_long_loc['Latitude'])

        min_long_loc = long_lat_df.loc[[long_lat_df['Longitude'].idxmin()]]
        min_long_long, min_long_lat = float(min_long_loc['Longitude']), float(min_long_loc['Latitude'])

        print(max_lat_lat, min_long_long)
        print(min_lat_lat, max_long_long)

        top_left = self.transformer.transform(min_long_long, max_lat_lat)
        bottom_right = self.transformer.transform(max_long_long, min_lat_lat)

        print("top_left", top_left)
        print("bottom_right", bottom_right)

        res = 10000
        top_left = (int((top_left[0]-res)/res)*res, math.ceil((top_left[1]+res)/res)*res)
        bottom_right = (math.ceil((bottom_right[0]+res)/res)*res, int((bottom_right[1]-res)/res)*res)

        print("top_left", top_left)
        print("bottom_right", bottom_right)
        """

        top_left = (-360000, 7240000)
        bottom_right = (1320000, 6030000)

        print("top_left", top_left)
        print("bottom_right", bottom_right)

        modulo = size % 2
        half_size = int(size / 2)

        print(self.resolution)

        """
        top_left = (top_left[0] - (top_left[0] % self.resolution), top_left[1] + (self.resolution - (top_left[1] % self.resolution)))
        bottom_right = (bottom_right[0] + (self.resolution - (bottom_right[0] % self.resolution)), bottom_right[1] - (bottom_right[1] % self.resolution))

        print(top_left)
        print(bottom_right)

        top_left = (top_left[0]-(half_size*self.resolution), top_left[1]+(half_size*self.resolution))
        bottom_right = (bottom_right[0]+((half_size+modulo)*self.resolution), bottom_right[1]-((half_size+modulo)*self.resolution))

        print(top_left)
        print(bottom_right)
        """


        print((bottom_right[0] - top_left[0]) / self.resolution)
        print((top_left[1] - bottom_right[1]) / self.resolution)

        list_raster = [lil_matrix((int((top_left[1] - bottom_right[1]) / self.resolution),
                       (int((bottom_right[0] - top_left[0]) / self.resolution))), dtype='uint8') for _ in range(3)]

        print(self.image_type)

        total = long_lat_df.shape[0]
        start = datetime.datetime.now()
        extract_time = 0

        modulo = size % 2
        half_size = int(size / 2)

        for idx, row in enumerate(long_lat_df.iterrows()):
            longitude, latitude = row[1][1], row[1][0]

            if idx % 10000 == 9999:
                _print_details(idx+1, total, start, extract_time, latitude, longitude, len(error_manager))

            t1 = ti.time()
            t2 = 0
            y, x = self.transformer.transform(longitude, latitude)
            try:
                patch = self.extract_patch_lambert93(x, y, size, step, white_percent_allowed=white_percent_allowed)
            except ExtractionError as err:
                t2 = ti.time()
                error_manager.append(err)
            else:
                t2 = ti.time()
                pos_x = int((top_left[1] - x)/self.resolution)
                pos_y = int((y - top_left[0])/self.resolution)
                for i in range(len(list_raster)):
                    list_raster[i][pos_x-half_size:pos_x+half_size+modulo, pos_y-half_size:pos_y+half_size+modulo] = patch[:, :, i]

            finally:
                delta = t2 - t1
                extract_time += delta
        error_manager.write_errors()
        for i in range(len(list_raster)):
            r = list_raster[i].tobsr()
            out = output_path(image_type_details[self.image_type][i]+".npz")
            print_info(image_type_details[self.image_type][i]+" channel saved at: "+out)
            save_npz(out, r)


class _ErrorManager(object):
    def __init__(self, in_proj, out_proj, path, cache_size=1000):
        # setting up the transformer
        self.transformer_out_in = Transformer.from_proj(out_proj, in_proj)

        # setting up the destination file
        current_datetime = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        error_extract_file = path + current_datetime + '_errors.csv'
        with open(error_extract_file, "w") as file:
            file.write(str(current_datetime) + "Occ_id;Error_type;Latitude;Longitude\n")
        self.file = error_extract_file

        # setting up the error cache
        self.error_cache = []
        self.total_size = 0

        # the maximum number of elements in the cache
        self.cache_size = cache_size

    def __len__(self):
        return self.total_size

    def append(self, error):
        """
        :param error:
        :return:
        """
        longitude, latitude = self.transformer_out_in.transform(error.y, error.x)

        self.error_cache.append('{};{};{};{}'.format(error.identifier, error.error_type, latitude, longitude))
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


def _print_details(idx, total, start, extract_time, latitude, longitude, nb_errors):
    time = datetime.datetime.now()
    print_info('\n{}/{}'.format(idx, total))
    p = ((idx - 1) / total) * 100
    print_info('%.2f' % p)
    delta = (time - start).total_seconds()
    estimation = (delta * total) / idx
    date_estimation = start + datetime.timedelta(seconds=estimation)
    print_info('mean extraction time: {}'.format(extract_time / idx))
    print_info('Actual position: {}, Errors: {}'.format((latitude, longitude), nb_errors))
    print_info('Time: {}, ETA: {}'.format(datetime.timedelta(seconds=delta), date_estimation))


if __name__ == "__main__":
    # im_manager = IGNImageManager("/data/ign/5M00/")
    # im_manager = IGNImageManager("/home/bdeneu/Desktop/IGN/BDORTHO_2-0_IRC-0M50_JP2-E080_LAMB93_D011_2015-01-01/")
    # im_manager = IGNImageManager("/home/bdeneu/Desktop/IGN/BDORTHO_2-0_RVB-0M50_JP2-E080_LAMB93_D011_2015-01-01_old")
    # im_manager = IGNImageManager("/home/data/5M00/")
    # im_manager = IGNImageManager("/home/bdeneu/Desktop/IGN/5M00/")  # "/home/bdeneu/Desktop/IGN/5M00/"
    # im_manager = IGNImageManager("/home/bdeneu/Desktop/IGN/0M50/")
    #im_manager = IGNImageManager("/home/bdeneu/Desktop/IGN/BDORTHO_2-0_IRC-0M50_JP2-E080_LAMB93_D011_2015-01-01/")
    im_manager = IGNImageManager("/home/bdeneu/data/ign/")

    np.set_printoptions(threshold=np.inf)
    print(im_manager.map)

    lat, long = 43.5972480773926, 3.88950347900391

    im = im_manager.get_image_at_location(lat, long)
    print(im.department, im.date, im.image_name)

    print(im_manager.map.shape)

    print(im_manager.image_type)

    try:
        im = im_manager.extract_patch(lat, long, 1.0, 256)
    except ExtractionError as err:
        print(err.error_type)

    np.save('/home/bdeneu/data/10068867285_ir.npy', im[:, :, 0])

    plt.imshow(im[:, :, 0])
    plt.show()
