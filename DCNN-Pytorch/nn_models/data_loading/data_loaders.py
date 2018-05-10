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
        self.images = np.tile(0, (self.length,3,im_size[0],im_size[1])).astype(np.int8)
        self.labels = np.tile(0, (len(self.annotations))).astype(np.float64)
        self.preloaded=False
    def statistics(self):
        mean = np.mean(self.images,(0,2,3))
        stdev = np.std(self.images,(0,2,3))
        return tuple(mean),tuple(stdev)
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
class F1SequenceDataset(F1Dataset):
    def __init__(self, root_folder, annotation_filepath, im_size,\
        context_length = 25, sequence_length=25, use_float32=False, img_transformation = None, label_transformation = None, optical_flow = False):
        super(F1SequenceDataset, self).__init__(root_folder, annotation_filepath, im_size, use_float32=use_float32, img_transformation = img_transformation, label_transformation = label_transformation)
        self.sequence_length = sequence_length
        self.context_length = context_length
        self.length -= (context_length + sequence_length)
        self.optical_flow=optical_flow
        if self.optical_flow:
            self.length -= 1

    def __getitem__(self, index):
        if(self.preloaded):  
            if not self.optical_flow:
                label_start = index + self.context_length
                label_end = label_start + self.sequence_length
                previous_control = self.labels[index:label_start]   
                seq = self.images[index:label_start]
                seq_labels = self.labels[label_start:label_end]
            else:
                label_start = index + self.context_length
                label_end = label_start + self.sequence_length
                previous_control = self.labels[index:label_start]
                seq_labels = self.labels[label_start:label_end]
                images = self.images[index:label_start]
                seq = np.random.rand(self.context_length,2,self.im_size[0],self.im_size[1])
                i = 0
                for idx in range(index, label_start):
                    color = self.images[idx].transpose(1,2,0).astype(np.float32)
                    prvs = cv2.cvtColor(color,cv2.COLOR_BGR2GRAY)
                    color = self.images[idx+1].transpose(1,2,0).astype(np.float32)
                    next = cv2.cvtColor(color,cv2.COLOR_BGR2GRAY)
                    seq[i] = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 20, 10, 7, 1.2, 0).transpose(2, 0, 1)
                    i+=1  
        else:
            raise NotImplementedError("Must preload images for sequence dataset")
        if(self.use_float32):
            seq = seq.astype(np.float32)
            seq_labels = seq_labels.astype(np.float32)
            previous_control = previous_control.astype(np.float32)
        else:
            seq = seq.astype(np.float64)
            seq_labels = seq_labels.astype(np.float64)
            previous_control = previous_control.astype(np.float64)
        label_tensor = torch.from_numpy(seq_labels)
        previous_control_tensor = torch.from_numpy(previous_control)
        img_tensor = torch.from_numpy(seq)
        if(not (self.img_transformation == None)):
            for i in range(0, img_tensor.shape[0]):
                img_tensor[i]=self.img_transformation(img_tensor[i])
        if(not (self.label_transformation == None)):
            for i in range(0, label_tensor.shape[0]):
                label_tensor[i]=self.label_transformation(label_tensor[i])
                previous_control_tensor[i]=self.label_transformation(previous_control_tensor[i])
        return img_tensor, previous_control_tensor.view(self.context_length,1), label_tensor.view(self.sequence_length,1)
    def __len__(self):
        return self.length