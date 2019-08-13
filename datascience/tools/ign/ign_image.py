import os
import numpy as np
from pyproj import Proj, transform
import matplotlib.pyplot as plt
import datetime

class Tuile(object):
    """
    Objet contenant les informations d'une tuile/une image

    liste attributs :
    - imaqe_name : le chemin d'accès depuis la raçine à la tuile correspondante
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
        self.x_min = int(data_value[2]) * 1000
        self.y_max = int(data_value[3]) * 1000
        self.date = str("".join(self.decomposed_name[-3].split("_")[-1].split("-")[0:2]))

    def setYmin(self,y_min):
        self.y_min = y_min

    def setXmax(self,x_max):
        self.x_max = x_max

    def __str__(self):
        return "Tuile("+str(self.department) + "_" + str(self.x_min) + "_" + str(self.y_max) + "_" + str(self.date)+")"

    def __repr__(self):
        return self.__str__()


class IGNImageManager(object):

    """
    Objet permetant la gestion des image IGN

    liste attibuts :
    - imdir : le chemin d'accès au dossier contenant toutes les miages de la zones détude. Les images doivent toutes être du même type (RVB/IR et et même résolution)
    - list_im : la liste des Tuiles de toutes la zone d'étude
    - min_x/max_x : les minimum/maximum est-ouest (croissant vers l'est) couvert par la zone détude
    - min_y/max_y : les minimum/maximum nord-sud (croissant vers le nord) couvert par la zone détude
    - image_resolution : résolution des images en m/pixels
    - image_type : le type d'image, RVB ou IR
    - image_encoding : l'encodage des images
    - image_projection : la projection utilisée pour les images (normalement LA93)
    - image_range : le rang de valeurs de projection couvert par une tuile
    - image_size : le nb de pixels en largeur et longeur d'une tuile
    - carto : matrice contenant les tuiles ordonnées par leur position relative
    """

    def __init__(self, imdir):
        self.error_extract_file = '/gpfsssd/scratch/rech/fqg/uid61lx/data/ign_5m_patches/error.csv'
        self.nb_errors = 0
        self.error_cache = []
        self.error_cache_size = 1000

        self.white_percent_allowed = 20

        self.imdir = imdir

        self.cache_dic = {}
        self.cache_list = []
        self.max_cahe = 1000

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
        self.list_im = []
        init = False
        for item in construction_list:
            if item.endswith(".jp2") or item.endswith(".tif"):
                tuile = Tuile(self.imdir+item)
                if not init:
                    init = True
                    self.min_x = tuile.x_min
                    self.max_x = tuile.x_min
                    self.min_y = tuile.y_max
                    self.max_y = tuile.y_max
                else:
                    if tuile.x_min < self.min_x:
                        self.min_x = tuile.x_min
                    elif tuile.x_min > self.max_x:
                        self.max_x = tuile.x_min
                    if tuile.y_max < self.min_y:
                        self.min_y = tuile.y_max
                    elif tuile.y_max > self.max_y:
                        self.max_y = tuile.y_max
                self.list_im.append(tuile)
        print(str(len(self.list_im)) + ' tuiles loaded...')
        for item in self.list_im:
            if item.image_name.endswith(".jp2") or item.image_name.endswith(".tif"):
                infos_images = item.decomposed_name[-2].split("_")
                self.image_resolution = infos_images[2]
                self.image_type = infos_images[1]
                self.image_encoding = infos_images[3]
                self.image_projection = infos_images[4]
                break

        if self.image_resolution == "5M00":
            self.image_range = 10000
            self.image_size = 2000
        elif self.image_resolution == "0M50":
            self.image_range = 5000
            self.image_size = 10000
        self.max_x = self.max_x + self.image_range
        self.min_y = self.min_y - self.image_range

        self.x_size = int((self.max_x-self.min_x)/self.image_range)
        self.y_size = int((self.max_y-self.min_y)/self.image_range)

        self.carto = np.empty((self.x_size, self.y_size), dtype=object)

        for tuile in self.list_im:
            tuile.setXmax(tuile.x_min+self.image_range)
            tuile.setYmin(tuile.y_max-self.image_range)
            pos_x, pos_y = self.convertToIndex(tuile.x_min, tuile.y_max)
            if type(self.carto[pos_x,pos_y]) != Tuile or int(self.carto[pos_x,pos_y].date)<int(tuile.date):
                self.carto[pos_x,pos_y] = tuile

    def init_error_file(self):
        currentDT = datetime.datetime.now()
        with open(self.error_extract_file, "a") as myfile:
            myfile.write(str(currentDT) + "\nOcc_id;Error_type;Latitude;Longitude\n")

    def getImdir(self):
        return self.imdir

    def getImList(self):
        return self.list_im

    def convertToIndex(self,x_in,y_in):
        x_out = int((x_in-self.min_x)/self.image_range)
        y_out = int((self.max_y-y_in)/self.image_range)
        return x_out,y_out

    def getImageAtLocation(self,lat,long):
        inProj = Proj(init='epsg:4326')
        outProj = Proj(init='epsg:2154')
        x1, y1 = long, lat
        x2, y2 = transform(inProj, outProj, x1, y1)
        # x2,y2 = int(x2),int(y2)
        x, y = self.convertToIndex(x2, y2)
        if x<0 or x>self.x_size-1 or y<0 or y>self.y_size-1 or type(self.carto[x,y])!=Tuile:
            print("hors de la zone d'étude")
            return
        return self.carto[x,y]

    def readImage(self, pos):
        if pos in self.cache_dic:
            im = self.cache_dic[pos]
        else:
            im = plt.imread(self.carto[pos[0],pos[1]].image_name)
            if len(self.cache_list)>=self.max_cahe:
                self.cache_dic.pop(self.cache_list[0])
                self.cache_list.pop(0)
                self.cache_list.append(pos)
                self.cache_dic[pos] = im
            else:
                self.cache_list.append(pos)
                self.cache_dic[pos] = im
        return im

    def extractPatchLatLong(self, lat, long, size, step, id=-1):
        inProj = Proj(init='epsg:4326')
        outProj = Proj(init='epsg:2154')
        x1, y1 = long, lat
        x2, y2 = transform(inProj, outProj, x1, y1)
        return self.extractPatch(x2, y2, size, step, id)

    def erreur_extract(self, x2, y2, type=1, id=-1):
        """
        :param x2:
        :param y2:
        :param type: 1 = partially outside of the area; 2 = center outside of the area; 3 = on the area but out of data coverage
        :param id:
        :return:
        """
        inProj = Proj(init='epsg:2154')
        outProj = Proj(init='epsg:4326')
        x1, y1 = x2, y2
        long, lat = transform(inProj, outProj, x1, y1)
        if type == 2:
            occurence = str(id) + ";T;" + str(lat) + ";" + str(long)
        elif type == 1:
            occurence = str(id) + ";P;" + str(lat) + ";" + str(long)
        elif type == 3:
            occurence = str(id) + ";I;" + str(lat) + ";" + str(long)
        self.error_cache.append(occurence)
        self.nb_errors += 1
        if len(self.error_cache) >= self.error_cache_size:
            self.write_errors()

    def write_errors(self):
        with open(self.error_extract_file, "a") as myfile:
            for occurence in self.error_cache:
                myfile.write(occurence + "\n")
        self.error_cache = []

    def extractPatch(self, x2, y2, size, step, id=-1):
        if step*size > self.image_size:
            print("sortie demandée trop grande")
            return
        x,y = self.convertToIndex(x2,y2)
        if x<0 or x>self.x_size-1 or y<0 or y>self.y_size-1 or type(self.carto[x,y])!=Tuile:
            self.erreur_extract(x2, y2, type=2, id=id)
            return

        im = self.readImage((x,y))

        pixel_x = int((x2-self.carto[x,y].x_min) * self.image_size / float(self.image_range))
        pixel_y = int((self.carto[x,y].y_max-y2) * self.image_size / float(self.image_range))

        modulo = size % 2
        half_size = int(size / 2)

        x_min = pixel_x - half_size * step
        x_max = pixel_x + half_size * step + modulo
        y_min = pixel_y - half_size * step
        y_max = pixel_y + half_size * step + modulo

        direction = 0

        if x_min < 0:

            x_max = x_max + self.image_size
            x_min = x_min + self.image_size
            direction = -1
            if x-1<0 or type(self.carto[x-1,y])!=Tuile:
                self.erreur_extract(x2, y2, id=id)
                return
            im2_name = self.carto[x-1,y].image_name
            im2 = self.readImage((x-1,y))
            im = np.concatenate((im2, im), axis=1)

        elif x_max > self.image_size - 1:

            direction = 1
            if x+1>self.x_size-1 or type(self.carto[x+1,y])!=Tuile:
                self.erreur_extract(x2, y2, id=id)
                return
            im2_name = self.carto[x+1,y].image_name
            im2 = self.readImage((x+1,y))
            im = np.concatenate((im, im2), axis=1)

        if y_min < 0:

            y_max = y_max + self.image_size
            y_min = y_min + self.image_size

            if direction == 0:
                if y - 1 < 0 or type(self.carto[x, y - 1]) != Tuile:
                    self.erreur_extract(x2, y2, id=id)
                    return
                im2_name = self.carto[x,y-1].image_name
                im2 = self.readImage((x,y-1))
                im = np.concatenate((im2, im), axis=0)

            elif direction == 1:
                if y - 1 < 0 or type(self.carto[x, y-1]) != Tuile or x+1>self.x_size-1 or type(self.carto[x+1,y-1]) != Tuile:
                    self.erreur_extract(x2, y2, id=id)
                    return
                im3_name = self.carto[x+1,y-1].image_name
                im4_name = self.carto[x,y-1].image_name
                im3 = self.readImage((x+1,y-1))
                im4 = self.readImage((x,y-1))
                im2 = np.concatenate((im4, im3), axis=1)
                im = np.concatenate((im2, im), axis=0)

            elif direction == -1:
                if y - 1 < 0 or type(self.carto[x, y-1]) != Tuile or x-1<0 or type(self.carto[x-1,y-1]) != Tuile:
                    self.erreur_extract(x2, y2, id=id)
                    return
                im3_name = self.carto[x-1,y-1].image_name
                im4_name = self.carto[x,y-1].image_name
                im3 = self.readImage((x-1,y-1))
                im4 = self.readImage((x,y-1))
                im2 = np.concatenate((im3, im4), axis=1)
                im = np.concatenate((im2, im), axis=0)

        elif y_max > self.image_size - 1:

            if direction == 0:
                if y+1 > self.image_size-1 or type(self.carto[x, y + 1]) != Tuile:
                    self.erreur_extract(x2, y2, id=id)
                    return
                im2_name = self.carto[x,y+1].image_name
                im2 = self.readImage((x,y+1))
                im = np.concatenate((im, im2), axis=0)

            elif direction == 1:
                if y+1>self.image_size or type(self.carto[x, y+1]) != Tuile or x+1>self.x_size-1 or type(self.carto[x+1,y+1]) != Tuile:
                    self.erreur_extract(x2, y2, id=id)
                    return
                im3_name = self.carto[x+1,y+1].image_name
                im4_name = self.carto[x,y+1].image_name
                im3 = self.readImage((x+1,y+1))
                im4 = self.readImage((x,y+1))
                im2 = np.concatenate((im4, im3), axis=1)
                im = np.concatenate((im, im2), axis=0)

            elif direction == -1:
                if y + 1 > self.image_size or type(self.carto[x, y+1]) != Tuile or x-1<0 or type(self.carto[x-1,y+1]) != Tuile:
                    self.erreur_extract(x2, y2, id=id)
                    return
                im3_name = self.carto[x-1,y+1].image_name
                im4_name = self.carto[x,y+1].image_name
                im3 = self.readImage((x-1,y+1))
                im4 = self.readImage((x,y+1))
                im2 = np.concatenate((im3, im4), axis=1)
                im = np.concatenate((im, im2), axis=0)

        rep = im[y_min:y_max:step, x_min:x_max:step, :]
        if (np.sum(np.all(rep == 255, axis=2))/(rep.shape[0]*rep.shape[1]))*100 > self.white_percent_allowed:
            self.erreur_extract(x2, y2, type=3, id=id)
            return
        return rep

if __name__ == "__main__":
    #im_manager = IGNImageManager("/data/ign/5M00/")
    #im_manager = IGNImageManager("/home/bdeneu/Desktop/IGN/BDORTHO_2-0_IRC-0M50_JP2-E080_LAMB93_D011_2015-01-01/")
    #im_manager = IGNImageManager("/home/bdeneu/Desktop/IGN/BDORTHO_2-0_RVB-0M50_JP2-E080_LAMB93_D011_2015-01-01_old")
    #im_manager = IGNImageManager("/home/data/5M00/")
    im_manager = IGNImageManager("/home/bdeneu/Desktop/IGN/5M00/")

    print(im_manager.carto)

    lat, long = 43.473090, 4.544300


    im = im_manager.getImageAtLocation(lat, long)
    print(im.department,im.date,im.image_name)

    print(im_manager.carto.shape)

    im = im_manager.extractPatchLatLong(lat, long, 64, 1)
    print(im)
    plt.imshow(im)
    plt.show()