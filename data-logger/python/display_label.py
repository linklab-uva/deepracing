import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("dataset_dir", help="Path to the directory of the dataset",  type=str)
parser.add_argument("oulabel_number", help="Directory to save the histograms to",  type=int)
args = parser.parse_args()
filepath = args.filepath
output_dir = args.output_dir
runtimes = np.loadtxt(filepath)

