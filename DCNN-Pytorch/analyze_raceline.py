import numpy.linalg as la
import numpy as np
import torch
from scipy.spatial.transform import Rotation as Rot
import matplotlib
from matplotlib import pyplot as plt
import argparse, argcomplete
import json

parser = argparse.ArgumentParser(description="Look at some statistical metrics of the optimal raceline")
parser.add_argument("racelinefile", type=str, help="Path to the raceline")
args = parser.parse_args()
argdict=vars(args)

racelinefile = argdict["racelinefile"]
with open(racelinefile, "r") as f:
    racelinedict = json.load(f)
raceline = np.column_stack([racelinedict[k] for k in ["x","y","z"]])
print(raceline)
