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
def pbImageToNpImage(im_pb : Image_pb2.Image):
    if not im_pb.channel_order == ChannelOrder_pb2.OPTICAL_FLOW:
        raise ValueError("Invalid channel order " + str(im_pb.channel_order) + " for optical flow dataset")
    return np.reshape(np.frombuffer(im_pb.image_data,dtype=np.float32),np.array((im_pb.rows, im_pb.cols, 2))).copy()
class OpticalFlowLMDBWrapper():
    def __init__(self, encoding = "ascii"):
        self.env = None
        self.encoding = encoding
        self.spare_txns=1
    def readImages(self, keys, db_path, image_wrapper, mapsize=int(1e9)):
        assert(len(keys) > 0)
        if os.path.isdir(db_path):
            raise IOError("Path " + db_path + " is already a directory")
        os.makedirs(db_path)
        env = lmdb.open(db_path, map_size=mapsize)
        print("Loading optical flow data")
        for i in tqdm(range(1,len(keys)), total=len(keys)-1):
            keyprev = keys[i-1]
            key = keys[i]
            img_prev = cv2.cvtColor( image_wrapper.getImage(keyprev) , cv2.COLOR_RGB2GRAY )
            img_curr = cv2.cvtColor( image_wrapper.getImage(key) , cv2.COLOR_RGB2GRAY )
            flow = cv2.calcOpticalFlowFarneback(img_prev, img_curr, None, 0.5, 3, 15, 3, 5, 1.2, 0).astype(np.float32)
            entry = Image_pb2.Image( rows=img_curr.shape[0] , cols=img_curr.shape[1] , channel_order=ChannelOrder_pb2.OPTICAL_FLOW , image_data=flow.flatten().tobytes() )
            with env.begin(write=True) as write_txn:
                write_txn.put(key.encode(self.encoding), entry.SerializeToString())
            with env.begin(write=False) as txn:
                im_pb = Image_pb2.Image()
                im_pb.ParseFromString( txn.get( key.encode( self.encoding ) ) )
                flow_from_db = pbImageToNpImage( im_pb )
                if not np.array_equal( flow , flow_from_db ):
                    print(flow)
                    print(flow_from_db)
                    raise ValueError("Database value is not the same as input value")
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
            im_pb.ParseFromString( txn.get( key.encode( self.encoding ) ) )
        return im_pb
    def getImage( self, key : str ):
        im_pb = self.getImagePB( key )
        return pbImageToNpImage( im_pb )
    def getNumImages(self):
        return self.env.stat()['entries']
    def getKeys(self):
        keys = None
        with self.env.begin(write=False) as txn:
            keys = [ str(key, encoding=self.encoding) for key, _ in txn.cursor() ]
        if (keys is None) or len(keys)==0:
            raise ValueError("Keyset is empty in optical flow dataset for some reason")
        return keys
