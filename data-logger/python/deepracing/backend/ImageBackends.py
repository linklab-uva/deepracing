from tqdm import tqdm as tqdm
import numpy as np
import skimage
import lmdb
import os
from skimage.transform import resize
import deepracing.imutils
import DeepF1_RPC_pb2_grpc
import DeepF1_RPC_pb2
import ChannelOrder_pb2
import Image_pb2
import grpc
import cv2
import time
import google.protobuf.empty_pb2 as Empty_pb2
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
class ImageGRPCClient():
    def __init__(self, address="127.0.0.1", port=50051):
        self.im_size = None
        self.channel = grpc.insecure_channel( "%s:%d" % ( address, port ) )
        self.stub = DeepF1_RPC_pb2_grpc.ImageServiceStub(self.channel)
    def getNumImages(self):
        response = self.stub.GetDbMetadata(DeepF1_RPC_pb2.DbMetadataRequest())
        return response.size
    def getImagePB(self, key):
        return self.stub.GetImage( DeepF1_RPC_pb2.ImageRequest(key=key) )
    def getImage(self, key):
        im_pb = self.getImagePB(key)
        return pbImageToNpImage(im_pb)
    def getKeys(self):
        response = self.stub.GetDbMetadata(DeepF1_RPC_pb2.DbMetadataRequest())
        return list(response.keys)
class ImageFolderWrapper():
    def __init__(self, image_folder):
        self.image_folder = image_folder
    def getImage( self, key : str ):
        fp = os.path.join(self.image_folder,key+".jpg")
        return deepracing.imutils.readImage(fp)
class ImageLMDBWrapper():
    def __init__(self, encoding = "ascii", direct_caching = False):
        self.env = None
        self.encoding = encoding
        self.spare_txns=1
        self.direct_caching = direct_caching
        self.internal_cache = {}
    def readImages(self, image_files, keys, db_path, im_size, ROI=None, mapsize=int(1e10)):
        assert(len(image_files) > 0)
        assert(len(image_files) == len(keys))
        if os.path.isdir(db_path):
            raise IOError("Path " + db_path + " is already a directory")
        os.makedirs(db_path)
        env = lmdb.open(db_path, map_size=mapsize)
        print("Loading image data")
        for i, key in tqdm(enumerate(keys), total=len(keys)):
            imgin = deepracing.imutils.readImage(image_files[i])
            if ROI is not None:
                x = ROI[0]
                y = ROI[1]
                w = ROI[2]
                h = ROI[3]
                imgin = imgin[y:y+h, x:x+w]
            im = deepracing.imutils.resizeImage( imgin , im_size[0:2] )
            entry = Image_pb2.Image( rows=im.shape[0] , cols=im.shape[1] , channel_order=ChannelOrder_pb2.RGB , image_data=im.flatten().tobytes() )
            with env.begin(write=True) as write_txn:
                write_txn.put(key.encode(self.encoding), entry.SerializeToString())
        env.close()
    def clearStaleReaders(self):
        self.env.reader_check()
    def resetEnv(self):
        if self.env is not None:
            path = self.env.path()
            mapsize = self.env.info()['map_size']
            self.env.close()
            del self.env
            time.sleep(1)
            self.readDatabase(path, mapsize=mapsize, max_spare_txns=self.spare_txns)
    def readDatabase(self, db_path : str, mapsize=int(1e10), max_spare_txns=1):
        if not os.path.isdir(db_path):
            raise IOError("Path " + db_path + " is not a directory")
        self.spare_txns = max_spare_txns
        self.env = lmdb.open(db_path, map_size=mapsize, readonly=True, max_spare_txns=max_spare_txns)#, lock=False)
    def getImagePB(self, key : str):
        im_pb = Image_pb2.Image()
        with self.env.begin(write=False) as txn:
            entry = txn.get( key.encode( self.encoding ) )
            if (entry is None):
                raise ValueError("Invalid key: %s on image database: %s" %(key, str(self.env.path())))
            im_pb.ParseFromString( entry )
        return im_pb
    def getImage( self, key : str ):
        if self.direct_caching:
            im = self.internal_cache.get(key,None)
            if im is not None:
                return im.copy()
            im_pb = self.getImagePB( key )
            im = pbImageToNpImage( im_pb )
            self.internal_cache["key"]=im
            return im.copy()
        else:  
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
