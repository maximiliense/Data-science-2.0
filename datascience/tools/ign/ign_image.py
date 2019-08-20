import os
import numpy as np
import matplotlib.pyplot as plt
import datetime
# from engine.logging.logs import print_logs
from pyproj import Transformer, Proj
import time as ti


class ExtractionError(Exception):
    pass


class Tile(object):
    """
    Objet contenant les informations d'une tile/une image

    liste attributs :
    - imaqe_name : le chemin d'accès depuis la raçine à la tile correspondante
    - decomposed_name : le chemin d'accès sous forme de liste (des dossier et su fichier)*
    - departement : le département présent sur la vue aérienne
    - date : la date de la vue aérienne
    - x_min/x_max : les minimum/maximum est-ouest (croissant vers l'est)
    - y_min/y_max : les minimum/maximum nord-sud (croissant vers le nord)
    """
    def __init__(self, image_name):
        self.image_name = image_name
        self.decomposed_name = image_name.split("/")
        data_value = self.decomposed_name[-1].split("-", 6)
        self.department = data_value[0]
        self.y_min = int(data_value[2]) * 1000
        self.x_max = int(data_value[3]) * 1000
        self.date = str("".join(self.decomposed_name[-3].split("_")[-1].split("-")[0:2]))

    def set_x_min(self, x_min):
        self.x_min = x_min

    def set_y_max(self, y_max):
        self.y_max = y_max

    def __str__(self):
        return "Tile("+str(self.department) + "_" + str(self.y_min) + "_" + str(self.x_max) + "_" + str(self.date)+")"

    def __repr__(self):
        return self.__str__()


class IGNImageManager(object):

    """
    Objet permetant la gestion des image IGN

    liste attibuts :
    - imdir : le chemin d'accès au dossier contenant toutes les miages de la zones détude. Les images doivent toutes être du même type (RVB/IR et et même résolution)
    - list_tile : la liste des tiles de toutes la zone d'étude
    - min_x/max_x : les minimum/maximum est-ouest (croissant vers l'est) couvert par la zone détude
    - min_y/max_y : les minimum/maximum nord-sud (croissant vers le nord) couvert par la zone détude
    - image_resolution : résolution des images en m/pixels
    - image_type : le type d'image, RVB ou IR
    - image_encoding : l'encodage des images
    - image_projection : la projection utilisée pour les images (normalement LA93)
    - image_range : le rang de valeurs de projection couvert par une tile
    - image_size : le nb de pixels en largeur et longeur d'une tile
    - carto : matrice contenant les tiles ordonnées par leur position relative
    """

    def __init__(self, imdir, max_cache=1000):
        self.imdir = imdir
        
        # config geographic projections
        self.proj_wgs84 = Proj(init='epsg:4326')
        self.proj_lambert93 = Proj(init='epsg:2154')
        self.transformer_wgs84_to_lambert93 = Transformer.from_proj(self.proj_wgs84, self.proj_lambert93)
        self.transformer_lambert93_to_wgs84 = Transformer.from_proj(self.proj_lambert93, self.proj_wgs84)

        # configure cache of tiles
        self.cache_dic = {}
        self.cache_list = []
        self.max_cahe = max_cache
        
        # get all files in the folder imdir
        construction_list = os.listdir(self.imdir)
        terminated = False
        while not terminated:
            terminated = True
            for i, file in enumerate(construction_list):
                if os.path.isdir(self.imdir+file):
                    folder = file+"/"
                    adding_list = [folder + s for s in os.listdir(self.imdir+folder)]
                    construction_list.remove(file)
                    construction_list.extend(adding_list)
                    terminated = False
        
        # construct tile object for all tile images in the list of files
        self.list_tile, self.min_x, self.max_x, self.min_y, self.max_y = self.get_tile_list(construction_list,
                                                                                            get_bounds=True)
        print(str(len(self.list_tile)) + ' tiles loaded !')
        
        # use the first tile in the list to get images information
        info_images = self.list_tile[0].decomposed_name[-2].split("_")
        self.image_resolution = info_images[2]
        self.image_type = info_images[1]
        self.image_encoding = info_images[3]
        self.image_projection = info_images[4]

        # size and range of images depends on the ign dataset
        if self.image_resolution == "5M00":
            self.image_range = 10000
            self.image_size = 2000
        elif self.image_resolution == "0M50":
            self.image_range = 5000
            self.image_size = 10000

        # configure the spatial area containing all tiles
        self.max_y = self.max_y + self.image_range
        self.min_x = self.min_x - self.image_range

        self.x_size = int((self.max_x-self.min_x)/self.image_range)
        self.y_size = int((self.max_y-self.min_y)/self.image_range)

        # create spatial matrix to order and place tiles relatively to their spatial position
        self.carto = np.empty((self.x_size, self.y_size), dtype=object)

        # fill the cartographic matrix with all tiles
        for tile in self.list_tile:
            tile.set_y_max(tile.y_min+self.image_range)
            tile.set_x_min(tile.x_max-self.image_range)
            pos_x, pos_y = self.position_to_tile_index(tile.x_max, tile.y_min)
            if type(self.carto[pos_x, pos_y]) != Tile or int(self.carto[pos_x, pos_y].date) < int(tile.date):
                # if 2 tiles cover the same area we keep the most recent one
                self.carto[pos_x, pos_y] = tile
    
    def get_tile_list(self, files_list, get_bounds):
        list_tile = []
        init = False
        for item in files_list:
            if item.endswith(".jp2") or item.endswith(".tif"):
                tile = Tile(self.imdir + item)
                if not init:
                    init = True
                    min_y = tile.y_min
                    max_y = tile.y_min
                    min_x = tile.x_max
                    max_x = tile.x_max
                else:
                    if tile.y_min < min_y:
                        min_y = tile.y_min
                    elif tile.y_min > max_y:
                        max_y = tile.y_min
                    if tile.x_max < min_x:
                        min_x = tile.x_max
                    elif tile.x_max > max_x:
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

    def init_error_file(self, error_extract_folder):
        currentDT = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        error_extract_file = error_extract_folder + currentDT + '_errors.csv'
        with open(error_extract_file, "w") as myfile:
            myfile.write(str(currentDT) + "Occ_id;Error_type;Latitude;Longitude\n")
        return error_extract_file

    def erreur_extract(self, x, y, error_type=1, id="NA"):
        """
        :param error_type:
        :param x:
        :param y:
        :param type: 1 = partially outside of the area; 2 = center outside of the area; 3 = on the area but out of data coverage
        :param id:
        :return:
        """
        long, lat = self.transformer_lambert93_to_wgs84.transform(y, x)
        err = str(id) + ";" + str(error_type) + ";" + str(lat) + ";" + str(long)
        return err

    def write_errors(self, error_extract_file, error_list):
        with open(error_extract_file, "a") as myfile:
            for err in error_list:
                myfile.write(err + "\n")

    def is_not_tile(self, x, y):
        if x < 0 or x > self.x_size - 1 or y < 0 or y > self.y_size - 1 or type(self.carto[x, y]) != Tile:
            return True
        return False

    def get_image_at_location(self, lat, long):
        y2, x2 = self.transformer_wgs84_to_lambert93.transform(long, lat)
        # x2,y2 = int(x2),int(y2)
        x, y = self.position_to_tile_index(x2, y2)
        if x < 0 or x > self.x_size-1 or y < 0 or y > self.y_size-1 or type(self.carto[x, y]) != Tile:
            print("hors de la zone d'étude")
            return
        return self.carto[x, y]

    def read_tile(self, pos):
        if pos in self.cache_dic:
            im = self.cache_dic[pos]
        else:
            im = plt.imread(self.carto[pos[0], pos[1]].image_name)
            if len(self.cache_list) >= self.max_cahe:
                self.cache_dic.pop(self.cache_list[0])
                self.cache_list.pop(0)
                self.cache_list.append(pos)
                self.cache_dic[pos] = im
            else:
                self.cache_list.append(pos)
                self.cache_dic[pos] = im
        return im

    def extract_patch_wgs84(self, lat, long, size, step, id=-1, white_percent_allowed=20, verbose=False):
        y, x = self.transformer_wgs84_to_lambert93.transform(long, lat)
        try:
            patch = self.extract_patch_lambert93(x, y, size, step, id, white_percent_allowed=white_percent_allowed,
                                                 verbose=verbose)
        except ExtractionError:
            raise
        return patch

    def extract_patch_lambert93(self, x_lamb, y_lamb, size, step, id=-1, white_percent_allowed=20, verbose=False):
        x, y = self.position_to_tile_index(x_lamb, y_lamb)

        if self.is_not_tile(x, y):
            raise ExtractionError(self.erreur_extract(x_lamb, y_lamb, error_type=2, id=id))

        # im = self.read_tile((x, y))

        pixel_y = int((y_lamb-self.carto[x, y].y_min) * self.image_size / float(self.image_range))
        pixel_x = int((self.carto[x, y].x_max-x_lamb) * self.image_size / float(self.image_range))

        modulo = size % 2
        half_size = int(size / 2)

        x_min = pixel_x - half_size * step
        x_max = pixel_x + half_size * step + modulo
        y_min = pixel_y - half_size * step
        y_max = pixel_y + half_size * step + modulo

        center_image_pos_x, center_image_pos_y = 0, 0
        aggregation_size_x, aggregation_size_y = 1, 1

        while x_min < 0 or x_max >= self.image_size*aggregation_size_x or\
                y_min < 0 or y_max >= self.image_size*aggregation_size_y:
            if x_min < 0:
                x_max = x_max + self.image_size
                x_min = x_min + self.image_size
                aggregation_size_x += 1
                center_image_pos_x += 1
            if y_min < 0:
                y_max = y_max + self.image_size
                y_min = y_min + self.image_size
                aggregation_size_y += 1
                center_image_pos_y += 1
            if x_max >= self.image_size*aggregation_size_x:
                aggregation_size_x += 1
            if y_max >= self.image_size*aggregation_size_y:
                aggregation_size_y += 1

        if verbose:
            print("number of tiles to load", aggregation_size_x*aggregation_size_y)

        list_im = []
        for i in range(aggregation_size_x):
            for j in range(aggregation_size_y):
                relative_x = i - center_image_pos_x
                relative_y = j - center_image_pos_y
                if self.is_not_tile(x + relative_x, y + relative_y):
                    raise ExtractionError(self.erreur_extract(x_lamb, y_lamb, error_type=1, id=id))
                list_im.append(((x + relative_x, y + relative_y),
                               (self.image_size*i, self.image_size*(i+1), self.image_size*j, self.image_size*(j+1))))

        aggregation_im = np.ndarray((self.image_size*aggregation_size_x, self.image_size*aggregation_size_y, 3),
                                    dtype=int)

        for im in list_im:
            aggregation_im[im[1][0]:im[1][1], im[1][2]:im[1][3]] = self.read_tile((im[0][0], im[0][1]))

        patch = aggregation_im[x_min:x_max:step, y_min:y_max:step, :]

        if (np.sum(np.all(patch == 255, axis=2))/(patch.shape[0]*patch.shape[1]))*100 > white_percent_allowed:
            raise ExtractionError(self.erreur_extract(x_lamb, y_lamb, error_type=3, id=id))
        return patch

    def extract_patches(self, df_occ, dest_dir, size=64, step=1, error_extract_folder=None,
                        error_cache_size=1000, white_percent_allowed=20, check_file=True, verbose=True):

        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

        nb_errors = 0
        error_cache = []
        if error_extract_folder is None:
            error_extract_file = self.init_error_file(dest_dir)
        else:
            error_extract_file = self.init_error_file(error_extract_folder)

        total = df_occ.shape[0]
        start = datetime.datetime.now()
        extract_time = 0

        for idx, row in enumerate(df_occ.iterrows()):
            long, lat = row[1][0], row[1][1]

            if (idx - 1) % 100000 == 99999 and verbose:
                time = datetime.datetime.now()
                print("\n" + str(idx) + "/" + str(total))
                p = ((idx - 1) / total) * 100
                print(str(p) + "%")
                delta = (time - start).total_seconds()
                estimation = (delta * total) / (idx + 1)
                date_estimation = start + datetime.timedelta(seconds=estimation)
                print("mean extraction time:", extract_time / idx)
                print("Actual position:", (lat, long), "Errors:", nb_errors)
                print("Time:", datetime.timedelta(seconds=delta), "ETA:", date_estimation)

            patch_id = int(row[1][2])

            # constructing path with hierarchical structure
            sub_d = dest_dir + str(patch_id)[-2:] + "/"
            if not os.path.exists(sub_d):
                os.makedirs(sub_d)
            sub_sub_d = sub_d + str(patch_id)[-4:-2] + "/"
            if not os.path.exists(sub_sub_d):
                os.makedirs(sub_sub_d)

            # if file exists pursue extraction
            if os.path.isfile(sub_sub_d + str(patch_id) + ".npy") and check_file:
                continue

            t1 = ti.time()
            t2 = 0
            try:
                patch = self.extract_patch_wgs84(lat, long, size, step, id=int(patch_id),
                                                 white_percent_allowed=white_percent_allowed)
            except ExtractionError as err:
                t2 = ti.time()
                nb_errors += 1
                error_cache.append(err)
                if len(error_cache) >= error_cache_size:
                    self.write_errors(error_extract_file, error_cache)
                    error_cache = []
            else:
                t2 = ti.time()
                np.save(sub_sub_d + str(patch_id) + ".npy", patch)
            finally:
                delta = t2 - t1
                extract_time += delta

        self.write_errors(error_extract_file, error_cache)


if __name__ == "__main__":
    # im_manager = IGNImageManager("/data/ign/5M00/")
    # im_manager = IGNImageManager("/home/bdeneu/Desktop/IGN/BDORTHO_2-0_IRC-0M50_JP2-E080_LAMB93_D011_2015-01-01/")
    # im_manager = IGNImageManager("/home/bdeneu/Desktop/IGN/BDORTHO_2-0_RVB-0M50_JP2-E080_LAMB93_D011_2015-01-01_old")
    # im_manager = IGNImageManager("/home/data/5M00/")
    im_manager = IGNImageManager("/home/bdeneu/Desktop/IGN/5M00/")

    print(im_manager.carto)

    lat, long = 46.665224, 2.543866

    im = im_manager.get_image_at_location(lat, long)
    print(im.department, im.date, im.image_name)

    print(im_manager.carto.shape)

    im = im_manager.extract_patch_wgs84(lat, long, 128, 30, verbose=True)
    print(im)
    plt.imshow(im)
    plt.show()
