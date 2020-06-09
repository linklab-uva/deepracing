from tqdm import tqdm as tqdm
import numpy as np
import skimage
import lmdb
import os
from skimage.transform import resize
import deepracing.imutils
import PoseSequenceLabel_pb2
import MultiAgentLabel_pb2
import ChannelOrder_pb2
import cv2
import google.protobuf.json_format
import google.protobuf.empty_pb2 as Empty_pb2
import time
import LabeledImage_pb2
import ImageLabel_pb2
class ControlLabelLMDBWrapper():
    def __init__(self):
        self.env = None
        self.encoding = "ascii"
    def writeControlLabel(self, key : str, label : LabeledImage_pb2.LabeledImage):
        with self.env.begin(write=True) as write_txn:
            write_txn.put(key.encode(self.encoding), label.SerializeToString())
    def readDatabase(self, db_path : str, mapsize=1e10, max_spare_txns=1, readonly=True):
        if not os.path.isdir(db_path):
            raise IOError("Path " + db_path + " is not a directory")
        if self.env is not None:
            self.env.close()
        self.env = lmdb.open(db_path, map_size=int(round(mapsize)), readonly=readonly, max_spare_txns=max_spare_txns)
        self.env.reader_check()
    def getControlLabel(self, key):
        rtn = LabeledImage_pb2.LabeledImage()
        with self.env.begin(write=False) as txn:
            entry_in = txn.get( key.encode( self.encoding ) )
            if (entry_in is None):
                raise ValueError("Invalid key on control label database: %s" %(key))
            rtn.ParseFromString(entry_in)
        return rtn
    def getNumLabels(self):
        return self.env.stat()['entries']
    def getKeys(self):
        keys = None
        with self.env.begin(write=False) as txn:
            keys = [ str(key, encoding=self.encoding) for key, _ in txn.cursor() ]
        if (keys is None) or len(keys)==0:
            raise ValueError("Keyset is empty in control label dataset for some reason")
        return keys

class MultiAgentLabelLMDBWrapper():
    def __init__(self):
        self.env = None
        self.encoding = "ascii"
        self.spare_txns=1
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
    def openDatabase(self, db_path : str, mapsize=1e10, max_spare_txns=125, readonly=True, lock=False):
        if not os.path.isdir(db_path):
            raise IOError("Path " + db_path + " is not a directory")
        self.spare_txns = max_spare_txns
        self.env = lmdb.open(db_path, map_size=round(mapsize,None), max_spare_txns=max_spare_txns,\
            create=False, lock=lock, readonly=readonly)
        self.env.reader_check()
    def writeMultiAgentLabel(self, key, entry):
        with self.env.begin(write=True) as txn:
            txn.put(key.encode( self.encoding ),entry.SerializeToString())
    def getMultiAgentLabel(self, key):
        rtn = MultiAgentLabel_pb2.MultiAgentLabel()
        with self.env.begin(write=False) as txn:
            entry_in = txn.get( key.encode( self.encoding ) )#.tobytes()
            if (entry_in is None):
                raise ValueError("Invalid key on label database: %s" %(key))
            rtn.ParseFromString(entry_in)
        return rtn
    def getNumLabels(self):
        return self.env.stat()['entries']
    def getKeys(self):
        keys = None
        with self.env.begin(write=False) as txn:
            keys = [ str(key, encoding=self.encoding) for key, _ in txn.cursor() ]
        if (keys is None) or len(keys)==0:
            raise ValueError("Keyset is empty in label dataset for some reason")
        return keys

class PoseSequenceLabelLMDBWrapper():
    def __init__(self):
        self.env = None
        self.encoding = "ascii"
        self.spare_txns=1
    def readLabelFiles(self, label_files, db_path, mapsize=1e10):
        assert(len(label_files) > 0)
        if os.path.isdir(db_path):
            raise IOError("Path " + db_path + " is already a directory")
        os.makedirs(db_path)
        env = lmdb.open(db_path, map_size=mapsize)
        print("Loading pose sequnce label data")
        for filepath in tqdm(label_files):
            with open(filepath) as f:
                json_in = f.read()
                label = PoseSequenceLabel_pb2.PoseSequenceLabel()
                google.protobuf.json_format.Parse(json_in , label) 
                key = os.path.splitext(label.image_tag.image_file)[0]
                with env.begin(write=True) as write_txn:
                    #entry = json_in.encode(self.encoding)
                   # print(entry)
                    write_txn.put(key.encode(self.encoding), label.SerializeToString())
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
    def readDatabase(self, db_path : str, mapsize=1e10, max_spare_txns=125, readonly=True, lock=False):
        if not os.path.isdir(db_path):
            raise IOError("Path " + db_path + " is not a directory")
        self.spare_txns = max_spare_txns
        self.env = lmdb.open(db_path, map_size=round(mapsize,None), max_spare_txns=max_spare_txns,\
            create=False, lock=lock, readonly=readonly)
        self.env.reader_check()
    def writePoseSequenceLabel(self, key, entry):
        with self.env.begin(write=True) as txn:
            txn.put(key.encode( self.encoding ),entry.SerializeToString())
    def getPoseSequenceLabel(self, key):
        rtn = PoseSequenceLabel_pb2.PoseSequenceLabel()
        with self.env.begin(write=False) as txn:
            entry_in = txn.get( key.encode( self.encoding ) )#.tobytes()
            if (entry_in is None):
                raise ValueError("Invalid key on label database: %s" %(key))
            rtn.ParseFromString(entry_in)
        return rtn
    def getNumLabels(self):
        return self.env.stat()['entries']
    def getKeys(self):
        keys = None
        with self.env.begin(write=False) as txn:
            keys = [ str(key, encoding=self.encoding) for key, _ in txn.cursor() ]
        if (keys is None) or len(keys)==0:
            raise ValueError("Keyset is empty in label dataset for some reason")
        return keys