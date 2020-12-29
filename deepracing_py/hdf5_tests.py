import h5py
import cv2
import numpy as np
import argparse
from tqdm import tqdm as tqdm
parser = argparse.ArgumentParser()
parser.add_argument("h5file", help="Path to the h5file to read",  type=str)

args = parser.parse_args()


hf5file = h5py.File(args.h5file, 'r')

image_dset : h5py.Dataset = hf5file["/images"]

try:
    cv2.namedWindow("Dataset Image")
except:
    pass
for i in tqdm(range(image_dset.shape[0])):
    im = image_dset[i]
    try:
        cv2.imshow("Dataset Image", im)
        cv2.waitKey(5)
    except:
        pass