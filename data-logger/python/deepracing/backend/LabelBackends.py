from tqdm import tqdm as tqdm
import numpy as np
import skimage
import lmdb
import os
from skimage.transform import resize
import deepracing.imutils
import DeepF1_RPC_pb2_grpc
import DeepF1_RPC_pb2
import PoseSequenceLabel_pb2
import ChannelOrder_pb2
import grpc
import cv2
import google.protobuf.json_format
import google.protobuf.empty_pb2 as Empty_pb2
import time
class PoseSequenceLabelGRPCClient():
    def __init__(self, address="127.0.0.1", port=50052):
        self.im_size = None
        self.channel = grpc.insecure_channel( "%s:%d" % ( address, port ) )
        self.stub = DeepF1_RPC_pb2_grpc.LabelServiceStub(self.channel)
    def getPoseSequenceLabel(self, key):
        response = self.stub.GetPoseSequenceLabel( DeepF1_RPC_pb2.PoseSequenceLabelRequest(key=key) )
        return response
    def getNumLabels(self):
        response = self.stub.GetDbMetadata(Empty_pb2.Empty())
        return response.size
    def getKeys(self):
        response = self.stub.GetDbMetadata(Empty_pb2.Empty())
        return list(response.keys)

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
    def readDatabase(self, db_path : str, mapsize=1e10, max_spare_txns=1):
        if not os.path.isdir(db_path):
            raise IOError("Path " + db_path + " is not a directory")
        self.spare_txns = max_spare_txns
        self.env = lmdb.open(db_path, map_size=mapsize, readonly=True, max_spare_txns=max_spare_txns)#, lock=False)
        self.env.reader_check()
    def getPoseSequenceLabel(self, key):
        rtn = PoseSequenceLabel_pb2.PoseSequenceLabel()
        with self.env.begin(write=False) as txn:
            entry_in = txn.get( key.encode( self.encoding ) )#.tobytes()
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