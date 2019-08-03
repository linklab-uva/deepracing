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
import grpc
import cv2
import google.protobuf.empty_pb2 as Empty_pb2
class ImageGRPCClient():
    def __init__(self, address="127.0.0.1", port=50051):
        self.im_size = None
        self.channel = grpc.insecure_channel( "%s:%d" % ( address, port ) )
        self.stub = DeepF1_RPC_pb2_grpc.ImageServiceStub(self.channel)
    def getNumImages(self):
        response = self.stub.GetDbMetadata(Empty_pb2.Empty())
        return response.size
    def getImage(self, key):
        response = self.stub.GetImage( DeepF1_RPC_pb2.ImageRequest(key=key) )
        imshape = np.array( (response.rows, response.cols, 3) )
        im = np.reshape( np.frombuffer( response.image_data, dtype=np.uint8 ) , imshape )
        if(response.channel_order == ChannelOrder_pb2.ChannelOrder.BGR):
            im = cv2.cvtColor(im,cv2.COLOR_BAYER_BGR2RGB)
        return im
        
class ImageLMDBWrapper():
    def __init__(self):
        self.env = None
        self.im_size = None
        self.size_type = np.uint16
        self.size_key = "imsize"
        self.num_images_key = "num_images"
        self.encoding = "ascii"
    def readImages(self, image_files, keys, db_path, im_size, func=None, mapsize=1e11):
        assert(len(image_files) > 0)
        assert(len(image_files) == len(keys))
        if os.path.isdir(db_path):
            raise IOError("Path " + db_path + " is already a directory")
        os.makedirs(db_path)
        self.env = lmdb.open(db_path, map_size=mapsize)
        self.im_size = im_size.astype(self.size_type)
        with self.env.begin(write=True) as write_txn:
            print("Loading image data")
            write_txn.put(self.size_key.encode(self.encoding), self.im_size.tobytes())
            write_txn.put(self.num_images_key.encode(self.encoding), str(len(keys)).encode(self.encoding))
            for i, key in tqdm(enumerate(keys), total=len(keys)):
                imgin = deepracing.imutils.readImage(image_files[i])
                if func is not None:
                    imgin = func(imgin)
                im = deepracing.imutils.resizeImage(imgin, self.im_size[0:2])
                write_txn.put(key.encode(self.encoding), im.flatten().tobytes())
        self.txn = self.env.begin(write=False)
    def readDatabase(self, db_path : str, mapsize=1e11):
        if not os.path.isdir(db_path):
            raise IOError("Path " + db_path + " is not a directory")
        self.env = lmdb.open(db_path, map_size=mapsize)
        with self.env.begin(write=False) as txn:
            self.im_size = np.frombuffer(txn.get(self.size_key.encode(self.encoding)), dtype=self.size_type)
    def getImage(self, key):
        im = None
        with self.env.begin(write=False) as txn:
            im = np.reshape(np.frombuffer(txn.get(key.encode(self.encoding)), dtype=np.uint8), self.im_size)
        return im
    def getNumImages(self):
        size = 0
        with self.env.begin(write=False) as txn:
            size = int(txn.get(self.num_images_key.encode(self.encoding)))
        return size
