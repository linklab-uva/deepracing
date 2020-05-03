import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("filepath", help="Path to txt file containing the runtimes",  type=str)
parser.add_argument("output_dir", help="Directory to save the histograms to",  type=str)
args = parser.parse_args()
filepath = args.filepath
output_dir = args.output_dir
runtimes = np.loadtxt(filepath)
os.makedirs(output_dir,exist_ok=True)
#runtimes = runtimes[runtimes<=.10]
mean = np.mean(runtimes)
stdev = np.std(runtimes)
print("Mean runtime: %f. Mean Frequency: %f" % (mean,1/mean))
print("Standard Deviation in runtime: %f" % stdev)
plt.hist(runtimes)
plt.xlabel("Runtime (seconds)")
plt.ylabel("Number of samples")
plt.savefig(os.path.join(output_dir,"runtime_histogram.png"))
plt.savefig(os.path.join(output_dir,"runtime_histogram.eps"))
plt.savefig(os.path.join(output_dir,"runtime_histogram.svg"))
plt.show()

