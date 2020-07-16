import os
import readline
import argparse
import numpy as np
from tqdm import tqdm as tqdm

parser = argparse.ArgumentParser("Split the key file for a dataset into two pieces, train and validation")
parser.add_argument('keyfile', type=str,  help='The file to split')
parser.add_argument('--valratio', type=float, default=0.1,  help='The portion ([0,1]) of the dataset to split into a separate validation set')

args = parser.parse_args()

argdict = vars(args)

keyfile = argdict["keyfile"]
valratio = argdict["valratio"]

keydir = os.path.dirname(keyfile)


with open(keyfile,"r") as f:
    allkeys = [k.replace("\n","") for k in f.readlines()]

totalkeys = len(allkeys)
valkey_indices = np.sort(np.random.choice(totalkeys, size=int(round(valratio*totalkeys)), replace=False))

valkeys = []
trainkeys = []
for (i,key) in enumerate(allkeys):
    if (valkey_indices.shape[0]>0) and i == valkey_indices[0]:
        valkeys.append(key)
        valkey_indices = valkey_indices[1:]
    else:
        trainkeys.append(key)


print("Val ratio: %f" % (len(valkeys)/totalkeys, ) )
print("Train ratio: %f" % (len(trainkeys)/totalkeys, ) )

with open(os.path.join(keydir,"trainkeys.txt"), "w") as f:
    f.writelines([k + "\n" for k in trainkeys])
with open(os.path.join(keydir,"valkeys.txt"), "w") as f:
    f.writelines([k + "\n" for k in valkeys])