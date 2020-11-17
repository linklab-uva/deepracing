from tqdm import tqdm as tqdm
import numpy as np
import skimage
import lmdb
import os
from skimage.transform import resize
import deepracing.imutils
import ChannelOrder_pb2
import Image_pb2
import cv2
import time
import google.protobuf.empty_pb2 as Empty_pb2
import yaml
import PIL, PIL.Image as PILImage
import torchvision, torchvision.transforms.functional as F
import torchvision.transforms as TF
def pbImageToNpImage(im_pb : Image_pb2.Image):
    im = None
    if im_pb.channel_order == ChannelOrder_pb2.BGR:
        im = np.reshape(np.frombuffer(im_pb.image_data,dtype=np.uint8),np.array((im_pb.rows, im_pb.cols, 3)))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    elif im_pb.channel_order == ChannelOrder_pb2.RGB:
        im = np.reshape(np.frombuffer(im_pb.image_data,dtype=np.uint8),np.array((im_pb.rows, im_pb.cols, 3)))
    elif im_pb.channel_order == ChannelOrder_pb2.GRAYSCALE:
        im = np.reshape(np.frombuffer(im_pb.image_data,dtype=np.uint8),np.array((im_pb.rows, im_pb.cols)))
    elif im_pb.channel_order == ChannelOrder_pb2.RGBA:
        im = np.reshape(np.frombuffer(im_pb.image_data,dtype=np.uint8),np.array((im_pb.rows, im_pb.cols, 4)))
    elif im_pb.channel_order == ChannelOrder_pb2.BGRA:
        im = np.reshape(np.frombuffer(im_pb.image_data,dtype=np.uint8),np.array((im_pb.rows, im_pb.cols, 4)))
        im = cv2.cvtColor(im, cv2.COLOR_BGRA2RGBA)
    else:
        raise ValueError("Unknown channel order: " + im_pb.channel_order)
    return im#.copy()
class ImageArrayBackend():
    def __init__(self, keyfile ):
        with open(keyfile,"r") as keyfile_:
            self.keys = keyfile_.readlines()
        for i in range(len(self.keys)):
            self.keys[i] = self.keys[i].replace("\n","")
            
        self.keymap = {self.keys[i]:i for i in range(len(self.keys))}
        self.images = None
    def loadImages(self, image_folder, ROI = None, imsize = (66,200) ):
        print("Loading images in %s" %(image_folder))
        self.images = np.zeros((len(self.keys),imsize[0],imsize[1],3), dtype=np.uint8)
        if ROI is not None:
            x = ROI[0]
            y = ROI[1]
            w = ROI[2]
            h = ROI[3]
        for i, key in tqdm(enumerate(self.keys), total=len(self.keys)):
            fp = os.path.join(image_folder,key+".jpg")
           # print(fp)
            imin = deepracing.imutils.readImage(fp)
            if ROI is not None:
                imin = imin[y:y+h,x:x+w,:]
            self.images[i] = deepracing.imutils.resizeImage(imin, imsize)
    def getImage( self, key : str ):
        return self.images[self.keymap[key]]
class ImageFolderWrapper():
    def __init__(self, image_folder):
        self.image_folder = image_folder
    def getImage( self, key : str ):
        fp = os.path.join(self.image_folder,key+".jpg")
        return deepracing.imutils.readImage(fp)
class ImageLMDBWrapper():
    def __init__(self, encoding = "ascii"):
        self.env = None
        self.encoding = encoding
        self.spare_txns=1
        self.internal_cache = {}
    def readImages(self, image_files, keys, db_path, im_size, ROI=None, mapsize=int(1e10)):
        assert(len(image_files) > 0)
        assert(len(image_files) == len(keys))
        if os.path.isdir(db_path):
            raise IOError("Path " + db_path + " is already a directory")
        os.makedirs(db_path)
        if ROI is not None:
            cfgout = {"ROI":list(ROI)}
            yaml.dump(cfgout,open(os.path.join(db_path,"config.yaml"),"w"),Dumper=yaml.SafeDumper)
        env = lmdb.open(db_path, map_size=mapsize)
        print("Loading image data")
       # topil = TF.ToPILImage()
        cropped_images_dir = os.path.join(os.path.dirname(db_path),"cropped_images")
        os.makedirs(cropped_images_dir,exist_ok=True)
        for i, key in tqdm(enumerate(keys), total=len(keys)):
            imgin = deepracing.imutils.readImage(image_files[i])
            impil = F.to_pil_image(imgin)
            if bool(ROI):
                x = ROI[0]
                y = ROI[1]
                w = ROI[2]
                h = ROI[3]
                impilresize = F.resized_crop(impil,y,x,h,w,im_size,interpolation=PILImage.LANCZOS)
            else:
                impilresize = F.resize(impil,im_size, interpolation=PILImage.LANCZOS)
            impilresize.save(os.path.join(cropped_images_dir, "%s.jpg" % (key,)))
            im = np.asarray(impilresize)
            entry = Image_pb2.Image( rows=im.shape[0] , cols=im.shape[1] , channel_order=ChannelOrder_pb2.RGB , image_data=im.flatten().tobytes() )
            with env.begin(write=True) as write_txn:
                write_txn.put(key.encode(self.encoding), entry.SerializeToString())
        env.close()
    def writeImage(self, key, image):
        imarr = np.asarray(image)
        entry = Image_pb2.Image( rows=imarr.shape[0] , cols=imarr.shape[1] , channel_order=ChannelOrder_pb2.RGB , image_data=imarr.flatten().tobytes() )
        with self.env.begin(write=True) as write_txn:
            write_txn.put(key.encode(self.encoding), entry.SerializeToString())
        return entry
    def clearStaleReaders(self):
        self.env.reader_check()
    def resetEnv(self):
        if self.env is not None:
            path = self.env.path()
            mapsize = self.env.info()['map_size']
            self.env.close()
            del self.env
            time.sleep(1)
            self.readDatabase(path, mapsize=mapsize, max_spare_txns=self.spare_txns, readonly=True, lock=False)
    def readDatabase(self, db_path : str, mapsize=int(1e10), max_spare_txns=125, readonly=True, lock=False):
        if not os.path.isdir(db_path):
            raise IOError("Path " + db_path + " is not a directory")
        self.spare_txns = max_spare_txns
        self.env = lmdb.open(db_path, map_size=round(mapsize,None), max_spare_txns=max_spare_txns,\
            create=False, lock=lock, readonly=readonly)
    def getImagePB(self, key : str):
        im_pb = Image_pb2.Image()
        with self.env.begin(write=False) as txn:
            entry = txn.get( key.encode( self.encoding ) )
            if (entry is None):
                raise ValueError("Invalid key: %s on image database: %s" %(key, str(self.env.path())))
            im_pb.ParseFromString( entry )
        return im_pb
    def getImage( self, key : str ):
        return pbImageToNpImage( self.getImagePB( key ) )
    def getNumImages(self):
        return self.env.stat()['entries']
    def getKeys(self):
        keys = None
        with self.env.begin(write=False) as txn:
            keys = [ str(key, encoding=self.encoding) for key, _ in txn.cursor() ]
        if (keys is None) or len(keys)==0:
            raise ValueError("Keyset is empty in image dataset for some reason")
        return keys
