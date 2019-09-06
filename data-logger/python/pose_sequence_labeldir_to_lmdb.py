import numpy as np
import os
import argparse
import skimage
import skimage.io
import shutil
import deepracing.imutils
import deepracing.backend
import cv2
import random

def main():
    parser = argparse.ArgumentParser(description="Load an pose sequence label directory into a database")
    parser.add_argument("label_dir", type=str, help="Directory containing the labels")
    parser.add_argument("db_dir", type=str, help="Directory containing the LMDB")
    parser.add_argument("--mapsize", type=float, default=1e9, help="Map size for the LMDB.")
    args = parser.parse_args()
    label_dir = args.label_dir
    db_dir = args.db_dir
    files = [os.path.join(label_dir, fname) for fname in  os.listdir(label_dir) if os.path.isfile(os.path.join(label_dir,fname)) and os.path.splitext(fname)[1].lower()==".json"]
    mapsize = int(args.mapsize)
    db = deepracing.backend.PoseSequenceLabelLMDBWrapper()
    if(os.path.isdir(db_dir)):
      shutil.rmtree(db_dir)
    db.readLabelFiles(files, db_dir, mapsize=mapsize )
    db.readDatabase(db_dir, mapsize=mapsize, max_spare_txns=16 )
    keys = db.getKeys()
    idx = random.randint(0,len(keys)-1)
    print(db.getPoseSequenceLabel(keys[idx]))
if __name__ == '__main__':
  main()
