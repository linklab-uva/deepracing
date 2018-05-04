import torch
import torch.random
from torch.utils.data.dataset import Dataset
import os
from data_loading.image_loading import load_image
from tqdm import tqdm as tqdm
import pickle
class F1Dataset(Dataset):
    def __init__(self, root_folder, annotation_filepath, im_size):
        super(F1Dataset, self).__init__()
        self.im_size=im_size
        self.label_size = 1
        self.root_folder = root_folder
        self.annotation_filepath = annotation_filepath
        self.annotations_file = open(os.path.join(self.root_folder,self.annotation_filepath), "r")
        self.annotations = self.annotations_file.readlines()
        self.length = len(self.annotations)
        self.images = None
        self.labels = None
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
    def read_files(self, use_float32 = False):
        print("loading data")
        if(use_float32):
            self.images = torch.empty(self.length, self.im_size[0], self.im_size[1], self.im_size[2], dtype=torch.float)
            self.labels = torch.empty(self.length, self.label_size, dtype=torch.float)
        else:
            self.images = torch.empty(self.length, self.im_size[0], self.im_size[1], self.im_size[2], dtype=torch.double)
            self.labels = torch.empty(self.length, self.label_size, dtype=torch.double)
        for (idx,line) in tqdm(enumerate(self.annotations)):
            fp, ts, steering, throttle, brake = line.split(",")
            self.labels[idx][0] = float(steering)
            im = load_image(os.path.join(self.root_folder,"raw_images",fp),size=(self.im_size[1],self.im_size[2]),scale_factor=255.0, use_float32 = use_float32)
            self.images[idx]=torch.from_numpy(im)  
           
    def __getitem__(self, index):
        return self.images[index], self.labels[index]
    def __len__(self):
        return self.length
class F1AllControlDataset(Dataset):
    def __init__(self, root_folder, annotation_filepath, im_size):
        super(F1AllControlDataset, self).__init__()
        self.im_size=im_size
        self.label_size = 3
        self.root_folder = root_folder
        self.annotation_filepath = annotation_filepath
        self.annotations_file = open(os.path.join(self.root_folder,self.annotation_filepath), "r")
        self.annotations = self.annotations_file.readlines()
        self.length = len(self.annotations)
        self.images = None
        self.labels = None
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
    def read_files(self, use_float32 = False):
        print("loading data")
        if(use_float32):
            self.images = torch.empty(self.length, self.im_size[0], self.im_size[1], self.im_size[2], dtype=torch.float)
            self.labels = torch.empty(self.length, self.label_size, dtype=torch.float)
        else:
            self.images = torch.empty(self.length, self.im_size[0], self.im_size[1], self.im_size[2], dtype=torch.double)
            self.labels = torch.empty(self.length, self.label_size, dtype=torch.double)
        for (idx,line) in tqdm(enumerate(self.annotations)):
            fp, ts, steering, throttle, brake = line.split(",")
            self.labels[idx][0] = float(steering)
            self.labels[idx][1] = float(throttle)
            self.labels[idx][2] = float(brake)
            im = load_image(os.path.join(self.root_folder,"raw_images",fp),size=(self.im_size[1],self.im_size[2]),scale_factor=255.0, use_float32 = use_float32)
            self.images[idx]=torch.from_numpy(im)
    def __getitem__(self, index):
        return self.images[index], self.labels[index]
    def __len__(self):
        return self.length