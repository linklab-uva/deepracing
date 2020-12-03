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
import torch, torchvision, torchvision.transforms.functional as F
import PIL
import PIL.Image as Image

parser = argparse.ArgumentParser(description="Get PCA of an image dataset")
parser.add_argument("dataset_config_file", type=str,  help="Dataset Configuration file to load")
parser.add_argument("num_components", type=int,  help="Number of principle components to reduce to")
parser.add_argument("--sample_ratio", type=float, default=0.2 , help="Ratio of each dataset to sample")
parser.add_argument("--gpu", type=int, default=-1 , help="Use GPU to generate SVD")
parser.add_argument("--resize", type=float, default=1.0 , help="Scale the images up by this factor before display")
args = parser.parse_args()
argdict = vars(args)
dataset_config_file = argdict["dataset_config_file"]
num_components = argdict["num_components"]
gpu = argdict["gpu"]
resize = argdict["resize"]
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
        imagelist.append(F.to_tensor(image_wrapper.getImage(key).copy()).numpy().astype(np.float32))
sourcesize = imagelist[0].shape
flattened_image_array = np.array([im.flatten() for im in imagelist])
dataset_config_dir, dataset_config_basefile = os.path.split(dataset_config_file)
base_file_name = os.path.splitext(dataset_config_basefile)[0] + ("_pca_%d" % (num_components,))
print("Fitting a %d-component pca with %d samples" % (num_components, flattened_image_array.shape[0]))
if gpu>=0:
    flattened_image_torch = torch.from_numpy(flattened_image_array).cuda(gpu)
    flattened_image_torch.requires_grad=False
    flattened_image_stdevs, flattened_image_means = torch.std_mean(flattened_image_torch, dim = 0)
    q = min(num_components, flattened_image_torch.shape[0], flattened_image_torch.shape[1] )
    U, S, V = torch.pca_lowrank(flattened_image_torch, niter = 3, q=q, center=True)
    irand = int(np.random.randint(0, high=flattened_image_torch.shape[0], dtype=np.int64))

    improj = torch.matmul((flattened_image_torch[irand] - flattened_image_means).unsqueeze(0), V[:, :num_components])
    imrtreshape = (255.0*torch.clamp(torch.matmul(improj, V[:, :num_components].t())[0] + flattened_image_means, 0.0, 1.0)).cpu().numpy().astype(np.uint8).reshape(sourcesize).transpose(1,2,0)
    iminreshape = (255.0*flattened_image_torch[irand]).cpu().numpy().astype(np.uint8).reshape(sourcesize).transpose(1,2,0)
    explained_variances = ((S**2)/(flattened_image_torch.shape[0]-1)).cpu().numpy()
    explained_variance_ratios = explained_variances/np.sum(explained_variances)
    with open(os.path.join(dataset_config_dir, base_file_name + "_U.pt"), "wb") as f:
        torch.save(U, f)
    with open(os.path.join(dataset_config_dir, base_file_name + "_S.pt"), "wb") as f:
        torch.save(S, f)
    with open(os.path.join(dataset_config_dir, base_file_name + "_V.pt"), "wb") as f:
        torch.save(V, f)
    with open(os.path.join(dataset_config_dir, base_file_name + "_means.pt"), "wb") as f:
        torch.save(flattened_image_means, f)
    with open(os.path.join(dataset_config_dir, base_file_name + "_stdevs.pt"), "wb") as f:
        torch.save(flattened_image_stdevs, f)
else:
    pca = PCA(n_components=num_components, copy=True)
    pca.fit(flattened_image_array)
    with open(os.path.join(dataset_config_dir, base_file_name + ".pkl"), "wb") as f:
        pkl.dump(pca, f)

    explained_variance_ratios = np.array(pca.explained_variance_ratio_)
    explained_variances = np.array(pca.explained_variance_)

    irand = int(np.random.randint(0, high=flattened_image_array.shape[0], dtype=int))
    imin = flattened_image_array[irand]
    improj = np.matmul((imin - pca.mean_).reshape(1,-1), pca.components_.transpose())
    #improj = pca.transform(imin.reshape(1,-1))
    imrtreshape = (255.0*np.clip(pca.inverse_transform(improj), 0.0, 1.0))[0].astype(np.uint8).reshape(sourcesize).transpose(1,2,0)
    iminreshape = (255.0*imin).astype(np.uint8).reshape(sourcesize).transpose(1,2,0)
print("Done fitting")
dims = int(round(resize*torch.min(torch.FloatTensor(sourcesize[1:])).item()))
rtpil = Image.fromarray(imrtreshape)
rtpilresize = F.resize(rtpil, dims, interpolation=Image.LANCZOS)
inpil = Image.fromarray(iminreshape)
inpilresize = F.resize(inpil, dims, interpolation=Image.LANCZOS)


fig, (axratio, axval) = plt.subplots(nrows=1, ncols=2)

axratio.plot(np.arange(1, explained_variance_ratios.shape[0]+1, dtype=np.int32), explained_variance_ratios, label="Explained Variance Ratios")
axratio.set_title("Scree Plot (Ratios)")
axratio.set_xlabel("Number of Principle Components")
axratio.set_ylabel("Ratio of Explained Variance")
#axratio.legend()

axval.plot(np.arange(1, explained_variances.shape[0]+1, dtype=np.int32), explained_variances, label="Explained Variances")
axval.set_title("Scree Plot (Absolute Values)")
axval.set_xlabel("Number of Principle Components")
axval.set_ylabel("Explained Variance")
fig.suptitle("Scree plots for a %d-component PCA" % (num_components,))
#axval.legend()
plt.savefig(os.path.join(dataset_config_dir, base_file_name+"_scree.svg"))
plt.savefig(os.path.join(dataset_config_dir, base_file_name+"_scree.png"))
plt.savefig(os.path.join(dataset_config_dir, base_file_name+"_scree.eps"))
fig2, (axinput, axroundtrip) = plt.subplots(nrows=1, ncols=2)
axinput.imshow(np.array(inpilresize))
axinput.set_title("Original Image")
axroundtrip.imshow(np.array(rtpilresize))
axroundtrip.set_title("Round-Trip Projection")
fig2.suptitle("Example images for a %d-component PCA" % (num_components,))
plt.savefig(os.path.join(dataset_config_dir, base_file_name+"_example.png"))
plt.show()



