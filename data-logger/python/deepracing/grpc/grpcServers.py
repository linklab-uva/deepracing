import DeepF1_RPC_pb2_grpc
import DeepF1_RPC_pb2
import Image_pb2
import ChannelOrder_pb2
import grpc
import cv2
import numpy as np
import argparse
import skimage
import skimage.io as io
import os
import time
from concurrent import futures
import logging
import argparse
import lmdb
import cv2
import deepracing.backend
class ImageLMDBServer(DeepF1_RPC_pb2_grpc.ImageServiceServicer):
  def __init__(self, dbfolder : str, mapsize=int(1E10) , num_workers : int = 1):
    super(ImageLMDBServer, self).__init__()
    self.dbfolder=dbfolder
    self.backend = deepracing.backend.ImageLMDBWrapper()
    self.backend.readDatabase(self.dbfolder, mapsize=mapsize, max_spare_txns=num_workers)
  def GetImage(self, request, context):
    return self.backend.getImagePB(request.key)
  def GetDbMetadata(self, request, context):
    rtn =  DeepF1_RPC_pb2.DbMetadata()
    for key in self.backend.getKeys():
        rtn.keys.append(key)
    rtn.size = self.backend.getNumImages()
    return rtn
class PoseSequenceLabelLMDBServer(DeepF1_RPC_pb2_grpc.ImageServiceServicer):
  def __init__(self, dbfolder : str, mapsize = int(1e9), num_workers = 1):
    super(PoseSequenceLabelLMDBServer, self).__init__()
    self.dbfolder=dbfolder
    self.backend = deepracing.backend.PoseSequenceLabelLMDBWrapper()
    self.backend.readDatabase(self.dbfolder, mapsize=mapsize, max_spare_txns=num_workers)
  def GetPoseSequenceLabel(self, request, context):
    rtn =  self.backend.getPoseSequenceLabel(request.key)
    return rtn
  def GetDbMetadata(self, request, context):
    rtn =  DeepF1_RPC_pb2.DbMetadata()
    rtn.size = self.backend.getNumLabels()
    for key in self.backend.getKeys():
        rtn.keys.append(key)
    return rtn