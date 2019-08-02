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
_ONE_DAY_IN_SECONDS = 60 * 60 * 24
class ImageLMDBServer(DeepF1_RPC_pb2_grpc.ImageServiceServicer):
  def __init__(self, dbfolder : str):
    super(ImageLMDBServer, self).__init__()
    self.dbfolder=dbfolder
    self.size_key = "imsize"
    self.key_encoding = "ascii"
    self.lmdb_env = lmdb.open(dbfolder, map_size=1e11)
    with self.lmdb_env.begin(write=False) as txn:
        self.im_size = np.frombuffer(txn.get(self.size_key.encode(self.key_encoding)), dtype=np.uint16)
        print(self.im_size)
    #self.txn = self.lmdb_env.begin(write=False)
  def GetImage(self, request, context):
    rtn =  Image_pb2.Image()
    rtn.channel_order = ChannelOrder_pb2.ChannelOrder.RGB
    rtn.rows = self.im_size[0]
    rtn.cols = self.im_size[1]
    with self.lmdb_env.begin(write=False) as txn:
        rtn.image_data = txn.get(request.key.encode(self.key_encoding))
    return rtn
  def GetDbMetadata(self, request, context):
    rtn =  DeepF1_RPC_pb2.DbMetadata()
    with self.lmdb_env.begin(write=False) as txn:
        rtn.size = int(txn.get("num_images".encode(self.key_encoding)))
    return rtn

def serve():
    parser = argparse.ArgumentParser(description='Image server.')
    parser.add_argument('address', type=str)
    parser.add_argument('port', type=int)
    parser.add_argument('db_folder', type=str)
    parser.add_argument('--num_workers', type=int, default=1)
    args = parser.parse_args()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=args.num_workers))
    lmdbserver = ImageLMDBServer(args.db_folder)
    DeepF1_RPC_pb2_grpc.add_ImageServiceServicer_to_server(lmdbserver, server)
    server.add_insecure_port('%s:%d' % (args.address,args.port) )
    print("Starting server")
    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == '__main__':
    logging.basicConfig()
    serve()