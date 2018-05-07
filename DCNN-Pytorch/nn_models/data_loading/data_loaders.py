import torch
import torch.random
from torch.utils.data.dataset import Dataset
import os
from data_loading.image_loading import load_image
from tqdm import tqdm as tqdm
import pickle
import cv2
import numpy as np
import torchvision.transforms as transforms
class F1Dataset(Dataset):
    def __init__(self, root_folder, annotation_filepath, im_size, use_float32=False, img_transformation = None, label_transformation = None):
        super(F1Dataset, self).__init__()
        self.im_size=im_size
        self.label_size = 1
        self.use_float32=use_float32
        self.root_folder = root_folder
        self.img_transformation = img_transformation
        self.label_transformation = label_transformation
        self.annotation_filepath = annotation_filepath
        self.annotations_file = open(os.path.join(self.root_folder,self.annotation_filepath), "r")
        self.annotations = self.annotations_file.readlines()
        self.length = len(self.annotations)
        self.images = np.tile(0, (len(self.annotations),3,im_size[0],im_size[1])).astype(np.int8)
        self.labels = np.tile(0, (len(self.annotations))).astype(np.float64)
        self.preloaded=False
    def write_pickles(self,image_pickle, label_pickle):
        filename = image_pickle
        fp = open(filename, 'wb')
        pickle.dump(self.images, fp)
        fp.close()
        print('File %s is saved.' % filename)

        filename = label_pickle
        fp = open(filename, 'wb')
        pickle.dump(self.labels, fp)
        fp.close()
        print('File %s is saved.' % filename)
    def read_pickles(self,image_pickle, label_pickle):
        filename = image_pickle
        fp = open(filename, 'rb')
        self.images = pickle.load(fp)
        fp.close()

        filename = label_pickle
        fp = open(filename, 'rb')
        self.labels =  pickle.load(fp)
        fp.close()
        self.preloaded=True
    def read_files(self):
        print("loading data")
        for (idx,line) in tqdm(enumerate(self.annotations)):
            fp, ts, steering, throttle, brake = line.split(",")
            im = load_image(os.path.join(self.root_folder,"raw_images",fp))
            im = cv2.resize(im, (self.im_size[1], self.im_size[0]), interpolation = cv2.INTER_CUBIC)
            im = np.transpose(im, (2, 0, 1))
            self.images[idx] = im
            self.labels[idx] = float(steering)
        self.preloaded=True
    def __getitem__(self, index):
        if(self.preloaded):
            im = self.images[index]
            label = self.labels[index]
        else:
            fp, ts, steering, throttle, brake = self.annotations[index].split(",")
            im = load_image(os.path.join(self.root_folder,"raw_images",fp))
            im = cv2.resize(im, (self.im_size[1], self.im_size[0]), interpolation = cv2.INTER_CUBIC)
            im = np.transpose(im, (2, 0, 1))
            label = np.array((float(steering)))
        if(self.use_float32):
            im = im.astype(np.float32)
            label = label.astype(np.float32)
        else:
            im = im.astype(np.float64)
            label = label.astype(np.float64)
        label_tensor = torch.from_numpy(np.array(label))
        img_tensor = torch.from_numpy(im)
        if(not (self.img_transformation == None)):
            img_tensor = self.img_transformation(img_tensor)
        if(not (self.label_transformation == None)):
            label_tensor = self.label_transformation(label_tensor)
        return img_tensor, label_tensor.view(1)
    def __len__(self):
        return self.length
class F1SequenceDataset(Dataset):
    def __init__(self, root_folder, annotation_filepath, im_size,\
        context_length = 25, sequence_length=25, use_float32=False, img_transformation = None, label_transformation = None):
        super(F1SequenceDataset, self).__init__()
        self.im_size=im_size
        self.label_size = 1
        self.use_float32=use_float32
        self.root_folder = root_folder
        self.img_transformation = img_transformation
        self.label_transformation = label_transformation
        self.annotation_filepath = annotation_filepath
        self.annotations_file = open(os.path.join(self.root_folder,self.annotation_filepath), "r")
        self.annotations = self.annotations_file.readlines()
        self.sequence_length = sequence_length
        self.context_length = context_length
        self.length = len(self.annotations) - sequence_length - context_length
        self.images = np.tile(0, (len(self.annotations),3,im_size[0],im_size[1])).astype(np.int8)
        self.labels = np.tile(0, (len(self.annotations))).astype(np.float64)
        self.preloaded=False
    def write_pickles(self,image_pickle, label_pickle):
        filename = image_pickle
        fp = open(filename, 'wb')
        pickle.dump(self.images, fp)
        fp.close()
        print('File %s is saved.' % filename)

        filename = label_pickle
        fp = open(filename, 'wb')
        pickle.dump(self.labels, fp)
        fp.close()
        print('File %s is saved.' % filename)
    def read_pickles(self,image_pickle, label_pickle):
        filename = image_pickle
        fp = open(filename, 'rb')
        self.images = pickle.load(fp)
        fp.close()

        filename = label_pickle
        fp = open(filename, 'rb')
        self.labels =  pickle.load(fp)
        fp.close()
        self.preloaded=True
    def read_files(self):
        print("loading data")
        for (idx,line) in tqdm(enumerate(self.annotations)):
            fp, ts, steering, throttle, brake = line.split(",")
            im = load_image(os.path.join(self.root_folder,"raw_images",fp))
            im = cv2.resize(im, (self.im_size[1], self.im_size[0]), interpolation = cv2.INTER_CUBIC)
            im = np.transpose(im, (2, 0, 1))
            self.images[idx] = im
            self.labels[idx] = float(steering)
        self.preloaded=True
    def __getitem__(self, index):
        if(self.preloaded):
            seq = self.images[index:index+self.context_length]
            label_start = index+self.context_length
            seq_labels = self.labels[label_start:label_start+self.sequence_length]
        else:
            raise NotImplementedError("Must preload images for sequence dataset")
        if(self.use_float32):
            seq = seq.astype(np.float32)
            seq_labels = seq_labels.astype(np.float32)
        else:
            seq = seq.astype(np.float64)
            seq_labels = seq_labels.astype(np.float64)
        label_tensor = torch.from_numpy(seq_labels)
        img_tensor = torch.from_numpy(seq)
        if(not (self.img_transformation == None)):
            for i in range(0, img_tensor.shape[0]):
                img_tensor[i]=self.img_transformation(img_tensor[i])
        if(not (self.label_transformation == None)):
            for i in range(0, label_tensor.shape[0]):
                label_tensor[i]=self.label_transformation(label_tensor[i])
        return img_tensor, label_tensor.view(self.sequence_length,1)
    def __len__(self):
        return self.length