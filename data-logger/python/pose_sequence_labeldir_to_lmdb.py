import numpy as np
import os
import argparse
import skimage
import skimage.io
import shutil
import deepracing.imutils
import deepracing.backend
import cv2

def main():
    parser = argparse.ArgumentParser(description="Load an pose sequence label directory into a database")
    parser.add_argument("label_dir", type=str, help="Directory containing the labels")
    parser.add_argument("db_dir", type=str, help="Directory containing the LMDB")
    parser.add_argument("--mapsize", type=float, default=1e11, help="Map size for the LMDB.")
    args = parser.parse_args()
    label_dir = args.label_dir
    db_dir = args.db_dir
    files = [os.path.join(label_dir, fname) for fname in  os.listdir(label_dir) if os.path.isfile(os.path.join(label_dir,fname)) and os.path.splitext(fname)[1].lower()==".json"]
    db = deepracing.backend.PoseSequenceLabelLMDBWrapper()
    db.readLabelFiles(files, db_dir, mapsize=args.mapsize)
if __name__ == '__main__':
  main()
