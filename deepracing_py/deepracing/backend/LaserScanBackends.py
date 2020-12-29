from tqdm import tqdm as tqdm
import numpy as np
import skimage
import lmdb
import os
from skimage.transform import resize
import deepracing.imutils
import LaserScan_pb2
from LaserScan_pb2 import LaserScan
import ChannelOrder_pb2
import cv2
import google.protobuf.json_format
import google.protobuf.empty_pb2 as Empty_pb2
import time

class LaserScanLMDBWrapper():
    def __init__(self, encoding = "ascii"):
        self.env = None
        self.encoding = encoding
        self.spare_txns=1
    
    def openDatabase(self, db_path : str, mapsize=1e10, max_spare_txns=125, readonly=True, lock=False):
        if not os.path.isdir(db_path):
            raise IOError("Path " + db_path + " is not a directory")
        self.spare_txns = max_spare_txns
        self.env = lmdb.open(db_path, map_size=round(mapsize,None), max_spare_txns=max_spare_txns,\
            create=False, lock=lock, readonly=readonly)
        self.env.reader_check()

    def writeLaserScan(self, key, entry):
        with self.env.begin(write=True) as txn:
            txn.put(key.encode( self.encoding ),entry.SerializeToString())

    def getLaserScan(self, key):
        rtn = LaserScan()
        with self.env.begin(write=False) as txn:
            entry_in = txn.get( key.encode( self.encoding ) )#.tobytes()
            if (entry_in is None):
                raise ValueError("Invalid key on laser scan database: %s" %(key))
            rtn.ParseFromString(entry_in)
        return rtn

    def getNumScans(self):
        return self.env.stat()['entries']
        
    def getKeys(self):
        keys = None
        with self.env.begin(write=False) as txn:
            keys = [ str(key, encoding=self.encoding) for key, _ in txn.cursor() ]
        if (keys is None) or len(keys)==0:
            raise ValueError("Keyset is empty in laser scan dataset for some reason")
        return keys