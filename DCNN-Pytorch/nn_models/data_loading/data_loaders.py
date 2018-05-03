import torch
import torch.random
from torch.utils.data.dataset import Dataset
import os
from data_loading.image_loading import load_image
from tqdm import tqdm as tqdm
import pickle
class F1Dataset(Dataset):
    def __init__(self, root_folder, annotation_filepath, im_size, label_size):
        super(F1Dataset, self).__init__()
        self.im_size=im_size
        self.root_folder = root_folder
        self.annotation_filepath = annotation_filepath
        self.annotations_file = open(os.path.join(self.root_folder,self.annotation_filepath), "r")
        self.annotations = self.annotations_file.readlines()
        self.length = len(self.annotations)
        self.images = torch.empty(self.__len__(),im_size[0],im_size[1],im_size[2], dtype=torch.float)
        self.labels = torch.empty(self.__len__(),label_size, dtype=torch.float)
    def write_pickles(self,image_pickle, label_pickle):
        filename = image_pickle
        fp = open(filename, 'wb')
        pickle.dump(self.images.numpy(), fp)
        fp.close()
        print('File %s is saved.' % filename)

        filename = label_pickle
        fp = open(filename, 'wb')
        pickle.dump(self.labels.numpy(), fp)
        fp.close()
        print('File %s is saved.' % filename)
    def read_pickles(self,image_pickle, label_pickle):
        filename = image_pickle
        fp = open(filename, 'rb')
        self.images = torch.from_numpy(pickle.load(fp))
        fp.close()

        filename = label_pickle
        fp = open(filename, 'rb')
        self.labels =  torch.from_numpy(pickle.load(fp))
        fp.close()
    def read_files(self):
        idx = 0
        print("loading data")
        for line in tqdm(self.annotations):
            fp, ts, steering, throttle, brake = line.split(",")
            self.labels[idx].fill_(float(steering))
            im = load_image(os.path.join(self.root_folder,"raw_images",fp),size=(self.im_size[1],self.im_size[2]),scale_factor=255.0)
            self.images[idx]=torch.from_numpy(im)
            idx = idx + 1   
           
    def __getitem__(self, index):
        return self.images[index], self.labels[index]
    def __len__(self):
        return self.length