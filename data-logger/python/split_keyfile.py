import os
import readline
import argparse
import numpy as np
from tqdm import tqdm as tqdm

parser = argparse.ArgumentParser("Split the key file for a dataset into two pieces, train and validation")
parser.add_argument('keyfile', type=str,  help='The file to split')
parser.add_argument('--valratio', type=float, default=0.1,  help='The portion ([0,1]) of the dataset to split into a separate validation set')
parser.add_argument('--seqlength', type=int, default=5,  help='How long to make each sequence')

args = parser.parse_args()

argdict = vars(args)

keyfile = argdict["keyfile"]
valratio = argdict["valratio"]/2.0
trainratio = 1.0 - valratio
seqlength = argdict["seqlength"]

keydir = os.path.dirname(keyfile)


with open(keyfile,"r") as f:
    allkeys = [k.replace("\n","") for k in f.readlines()]

totalkeys = len(allkeys)
trainkey_indices = np.arange(0,totalkeys,step=1).tolist()
valkey_indices = np.sort(np.random.choice(trainkey_indices,size=int(round(valratio*totalkeys)), replace=False))
for vk in valkey_indices:
    for i in range(vk-seqlength+1, vk+seqlength):
        try:
            trainkey_indices.remove(i)
        except ValueError as e:
            pass

valkeys = [allkeys[i] for i in valkey_indices]
trainkeys = [allkeys[i] for i in trainkey_indices]

with open(os.path.join(keydir,"trainkeys.txt"), "w") as f:
    f.writelines([k + "\n" for k in trainkeys])
with open(os.path.join(keydir,"valkeys.txt"), "w") as f:
    f.writelines([k + "\n" for k in valkeys])