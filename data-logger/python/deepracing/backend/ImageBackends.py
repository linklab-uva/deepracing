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
    elif im_pb.channel_order == ChannelOrder_pb2.OPTICAL_FLOW:
        im = np.reshape(np.frombuffer(im_pb.image_data,dtype=np.float32),np.array((im_pb.rows, im_pb.cols, 2)))
    else:
        raise ValueError("Unknown channel order: " + im_pb.channel_order)
    return im
class ImageGRPCClient():
    def __init__(self, address="127.0.0.1", port=50051):
        self.im_size = None
        self.channel = grpc.insecure_channel( "%s:%d" % ( address, port ) )
        self.stub = DeepF1_RPC_pb2_grpc.ImageServiceStub(self.channel)
    def getNumImages(self):
        response = self.stub.GetDbMetadata(Empty_pb2.Empty())
        return response.size
    def getImagePB(self, key):
        return self.stub.GetImage( DeepF1_RPC_pb2.ImageRequest(key=key) )
    def getImage(self, key):
        im_pb = self.getImagePB(key)
        return pbImageToNpImage(im_pb)
        
class ImageLMDBWrapper():
    def __init__(self):
        self.env = None
        self.encoding = "ascii"
    def readImages(self, image_files, keys, db_path, im_size, func=None, mapsize=int(1e10)):
        assert(len(image_files) > 0)
        assert(len(image_files) == len(keys))
        if os.path.isdir(db_path):
            raise IOError("Path " + db_path + " is already a directory")
        os.makedirs(db_path)
        env = lmdb.open(db_path, map_size=mapsize)
        print("Loading image data")
        for i, key in tqdm(enumerate(keys), total=len(keys)):
            imgin = deepracing.imutils.readImage(image_files[i])
            if func is not None:
                imgin = func(imgin)
            im = deepracing.imutils.resizeImage( imgin , im_size[0:2] )
            entry = Image_pb2.Image( rows=im.shape[0] , cols=im.shape[1] , channel_order=ChannelOrder_pb2.RGB , image_data=im.flatten().tobytes() )
            with env.begin(write=True) as write_txn:
                write_txn.put(key.encode(self.encoding), entry.SerializeToString())
        env.close()
    def readDatabase(self, db_path : str, mapsize=int(1e10), max_spare_txns=1):
        if not os.path.isdir(db_path):
            raise IOError("Path " + db_path + " is not a directory")
        self.env = lmdb.open(db_path, map_size=mapsize, readonly=True, max_spare_txns=max_spare_txns)
    def getImagePB(self, key):
        im_pb = Image_pb2.Image()
        with self.env.begin(write=False, buffers=True) as txn:
            im_pb.ParseFromString(txn.get(key.encode(self.encoding)))
        return im_pb
    def getImage(self, key):
        im_pb = self.getImagePB(key)
        return pbImageToNpImage(im_pb)
    def getNumImages(self):
        return self.env.stat()['entries']
    def getKeys(self):
        keys = None
        with self.env.begin(write=False) as txn:
            keys = [ str(key, encoding=self.encoding) for key, _ in txn.cursor() ]
        if (keys is None) or len(keys)==0:
            raise ValueError("Keyset is empty in image dataset for some reason")
        return keys
