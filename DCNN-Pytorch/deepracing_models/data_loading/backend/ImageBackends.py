from tqdm import tqdm as tqdm
import numpy as np
import skimage
import lmdb
import os
from skimage.transform import resize
import imutils
import ImageRPC_pb2_grpc
import ImageRPC_pb2
import ChannelOrder_pb2
import grpc
class GRPCWrapper():
    def __init__(self, address="127.0.0.1", port=50051):
        self.im_size = None
        self.channel = grpc.insecure_channel( "%s:%d" % ( address, port ) )
        self.stub = ImageRPC_pb2_grpc.ImageServiceStub(self.channel)
    def getImage(self, key):
        response = self.stub.GetImage( ImageRPC_pb2.ImageRequest(key=key) )
        imshape = np.array( (response.image.rows, response.image.cols, 3) )
        return np.reshape( np.frombuffer( response.image.image_data, dtype=np.uint8 ) , imshape )
        
class LMDBWrapper():
    def __init__(self):
        self.txn = None
        self.env = None
        self.im_size = None
        self.size_type = np.uint16
        self.size_key = "imsize"
        self.key_encoding = "ascii"
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
            write_txn.put(self.size_key.encode(self.key_encoding), self.im_size.tobytes())
            for i, key in tqdm(enumerate(keys)):
                imgin = imutils.readImage(image_files[i])
                if func is not None:
                    imgin = func(imgin)
                im = imutils.resizeImage(imgin, self.im_size[0:2])
                write_txn.put(key.encode(self.key_encoding), im.flatten().tobytes())
        self.txn = self.env.begin(write=False)
    def readDatabase(self, db_path : str, mapsize=1e11):
        if not os.path.isdir(db_path):
            raise IOError("Path " + db_path + " is not a directory")
        self.env = lmdb.open(db_path, map_size=mapsize)
        self.txn = self.env.begin(write=False)
        self.im_size = np.fromstring(self.txn.get(self.size_key.encode(self.key_encoding)), dtype=self.size_type)
    def getImage(self, key):
        return np.reshape(np.fromstring(self.txn.get(key.encode(self.key_encoding)), dtype=np.uint8), self.im_size)
