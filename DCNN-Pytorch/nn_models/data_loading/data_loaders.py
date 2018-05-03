import torch
import torch.random
from torch.utils.data.dataset import Dataset
import os
from data_loading.image_loading import load_image
class F1Dataset(Dataset):
    def __init__(self, root_folder, annotation_filepath):
        super(F1Dataset, self).__init__()
        self.root_folder = root_folder
        self.annotation_filepath = annotation_filepath
        self.annotations_file = open(os.path.join(self.root_folder,self.annotation_filepath), "r")
        self.annotations = self.annotations_file.readlines()
    def __getitem__(self, index):
        # stuff
        fp, ts, steering, throttle, brake = self.annotations[index].split(",")
        label = torch.empty(1, dtype=torch.double)
        label.fill_(float(steering))
        im = load_image(os.path.join(self.root_folder,"raw_images",fp),size=(66,200),scale_factor=255.0)
        return torch.from_numpy(im), label
    def __len__(self):
        return len(self.annotations)