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

class PoseSequenceLabelLMDBWrapper():
    def __init__(self):
        self.env = None
        self.encoding = "ascii"
        self.num_labels_key = "num_labels"
    def readLabelFiles(self, label_files, db_path, mapsize=1e11):
        assert(len(label_files) > 0)
        if os.path.isdir(db_path):
            raise IOError("Path " + db_path + " is already a directory")
        os.makedirs(db_path)
        self.env = lmdb.open(db_path, map_size=mapsize)
        with self.env.begin(write=True) as write_txn:
            print("Loading pose sequnce label data")
            write_txn.put(self.num_labels_key.encode(self.encoding), str(len(label_files)).encode(self.encoding))
            for filepath in tqdm(label_files):
                with open(filepath) as f:
                    json_in = f.read()
                    label = PoseSequenceLabel_pb2.PoseSequenceLabel()
                    google.protobuf.json_format.Parse(json_in , label) 
                    key = os.path.splitext(label.image_tag.image_file)[0]
                    print(key)
                    write_txn.put(key.encode(self.encoding), json_in.encode(self.encoding))
    def readDatabase(self, db_path : str, mapsize=1e11):
        if not os.path.isdir(db_path):
            raise IOError("Path " + db_path + " is not a directory")
        self.env = lmdb.open(db_path, map_size=mapsize)
    def getPoseSequenceLabel(self, key):
        rtn = PoseSequenceLabel_pb2.PoseSequenceLabel()
        with self.env.begin(write=False) as txn:
            google.protobuf.json_format.Parse(txn.get(key.encode(self.encoding)).decode(self.encoding) , rtn) 
        return rtn
    def getNumLabels(self):
        size = 0
        with self.env.begin(write=False) as txn:
            size = int(txn.get(self.num_labels_key.encode(self.encoding)))
        return size