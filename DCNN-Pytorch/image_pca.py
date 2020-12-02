import deepracing
from deepracing.backend.ImageBackends import ImageLMDBWrapper, ImageFolderWrapper
import pickle as pkl
import yaml
import argparse
import os
import numpy as np
import random
import scipy, scipy.stats
import sklearn
from sklearn.decomposition import PCA, IncrementalPCA
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Get PCA of an image dataset")
parser.add_argument("dataset_config_file", type=str,  help="Dataset Configuration file to load")
parser.add_argument("num_components", type=int,  help="Number of principle components to reduce to")
parser.add_argument("--sample_ratio", type=float, default=0.2 , help="Ratio of each dataset to sample")
args = parser.parse_args()
argdict = vars(args)
dataset_config_file = argdict["dataset_config_file"]
num_components = argdict["num_components"]
sample_ratio = min(argdict["sample_ratio"], 1.0)
with open(dataset_config_file,"r") as f:
    dataset_config = yaml.load(f, Loader=yaml.SafeLoader)




image_size = dataset_config["image_size"]
wrappers = []
for dataset in dataset_config["datasets"]:
    print("Parsing database config: %s" %(str(dataset)))
    root_folder = dataset["root_folder"]
    image_folder = os.path.join(root_folder,"images")
    image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder,f)) and (os.path.splitext(f)[-1].lower() in {".jpg", ".png"})]
    image_mapsize = int(float(np.prod(image_size)*3+12)*float(len(image_files))*1.1)

    image_lmdb_folder = os.path.join(image_folder,"image_lmdb")
    image_wrapper = deepracing.backend.ImageLMDBWrapper()
    image_wrapper.readDatabase( image_lmdb_folder , mapsize=image_mapsize )
    keys = image_wrapper.getKeys()
    wrappers.append((keys, image_wrapper))

print("Grabbing %f%% of the images from each dataset " %(sample_ratio*100.0,))
imagelist = []
for (keys, image_wrapper) in wrappers:
    sample_keys = random.sample(keys, int(sample_ratio*len(keys)))
    for key in sample_keys:
        imagelist.append(image_wrapper.getImage(key))
flattened_image_array = np.array([im.flatten() for im in imagelist]).astype(np.float64)#/255.0

pca = IncrementalPCA(n_components=num_components, copy=True)
print("Fitting a %d-component pca with %d samples" % (num_components, flattened_image_array.shape[0]))
pca.fit(flattened_image_array)
print("Done fitting")

dataset_config_dir, dataset_config_basefile = os.path.split(dataset_config_file)
base_file_name = os.path.splitext(dataset_config_basefile)[0] + ("_pca_%d" % (num_components,))
with open(os.path.join(dataset_config_dir, base_file_name + ".pkl"), "wb") as f:
    pkl.dump(pca, f)

explained_variance_ratios = np.array(pca.explained_variance_ratio_)
explained_variances = np.array(pca.explained_variance_)
print(explained_variance_ratios)

sourcesize = imagelist[0].shape
imin = flattened_image_array[np.random.randint(0, high=flattened_image_array.shape[0], dtype=np.int64)]
imroundtrip = (pca.inverse_transform(pca.transform(imin.reshape(1,-1)))[0].reshape(sourcesize)).astype(np.uint8)
iminreshape = (imin.reshape(sourcesize)).astype(np.uint8)


fig, (axratio, axval, axinput, axroundtrip) = plt.subplots(nrows=1, ncols=4)

axratio.plot(np.arange(1, explained_variance_ratios.shape[0]+1, dtype=np.int32), explained_variance_ratios, label="Explained Variance Ratios")
axratio.set_title("Scree Plot (Ratios)")
axratio.set_xlabel("Number of Principle Components")
axratio.set_ylabel("Ratio of Explained Variance")
#axratio.legend()

axval.plot(np.arange(1, explained_variances.shape[0]+1, dtype=np.int32), explained_variances, label="Explained Variances")
axval.set_title("Scree Plot (Absolute Values)")
axval.set_xlabel("Number of Principle Components")
axval.set_ylabel("Ratio of Explained Variance")
#axval.legend()
axinput.imshow(iminreshape)
axroundtrip.imshow(imroundtrip)
plt.show()



