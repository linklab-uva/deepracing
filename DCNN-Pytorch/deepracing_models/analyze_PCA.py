import numpy.linalg as la
import numpy as np
import torch
import torch.utils.data as data_utils
import torchvision
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import matplotlib.animation as animation
def plot_images(batch, rows, cols, title = ""):
    plt.figure(figsize = (rows, cols))
    plt.axis("off")
    plt.title(title)
    plt.imshow(np.transpose(
        vutils.make_grid(batch[:(rows * cols)], nrow = rows, normalize = True).cpu(),
        (1, 2, 0)))
imchannels = 3
imrows = 66
imcols = 200
dataset_name = "control_labels"
with open(dataset_name+"_data_matrix.pt",'rb') as f:
    datamatrix_torch = torch.load(f)
# with open(dataset_name+"_covariance.pt",'rb') as f:
#     C_torch = torch.load(f)
with open(dataset_name+"_eigenvalues.pt",'rb') as f:
    eigenvalues_real = torch.load(f)
with open(dataset_name+"_eigenvectors.pt",'rb') as f:
    eigenvectors = torch.load(f)
num_images = 9
means = torch.mean(datamatrix_torch,0)
D = datamatrix_torch.shape[1]
comp = []
i = np.random.randint(0,high=datamatrix_torch.shape[0]-num_images-1)
print(i)
im = datamatrix_torch[i]
im_reconstruc = torch.zeros(num_images+1,D)
im_reconstruc[0] = im.clone()
for i in range(1,num_images+2):
    d = int(round(D/(2**i)))
    s = torch.diag(eigenvalues_real[0:d])
    pcs = eigenvectors[:,0:d]
    pcsT = pcs.transpose(0,1)
    proj = torch.matmul(im - means, pcs)
    unproj = torch.matmul(proj, pcsT)
    im_reconstruc[i-1] = (unproj + means).clone()
plot_images(im_reconstruc.view(num_images+1,imchannels, imrows, imcols), 1, num_images+1)
plt.show()