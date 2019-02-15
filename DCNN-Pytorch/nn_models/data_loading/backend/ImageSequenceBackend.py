import torch
import os
import PIL
from PIL import Image as PILImage
import torchvision.transforms as transforms
import torchvision
from tqdm import tqdm as tqdm
import numpy as np
import abc
from typing import List
import threading
import time
class DeepF1ImageSequenceBackend(metaclass=abc.ABCMeta):
    def __init__(self, context_length : int, sequence_length : int):
        self.context_length = context_length
        self.sequence_length = sequence_length

    @abc.abstractmethod
    def getImageRange(self, index : int):
        pass
    
    @abc.abstractmethod
    def getLabelRange(self, index : int):
        pass

    @abc.abstractmethod
    def numberOfImages(self):
        pass
class DeepF1LeaderFollowerBackend(DeepF1ImageSequenceBackend):
    def __init__(self, annotation_file : str, index_order: List[int], context_length : int, sequence_length : int, buffer_size : int):
        super(DeepF1LeaderFollowerBackend, self).__init__(context_length, sequence_length)
        self.annotation_file : str = annotation_file
        self.image_directory = os.path.join(os.path.dirname(annotation_file),'raw_images')
        f = open(self.annotation_file, 'r')
        self.annotations : List[str] = f.readlines()
        f.close()
        self.image_dict : dict = {}
        self.image_lock = threading.Lock()
        self.image_leader_index : int = 0
        self.label_dict : dict = {}
        self.label_lock = threading.Lock()
        self.label_leader_index : int = 0
        self.index_order : List[int] = index_order
        self.buffer_size = buffer_size
        print('Prefilling the buffer')
        self.resize = torchvision.transforms.Resize((66,200))
        self.totensor = torchvision.transforms.ToTensor()
        for i in tqdm(range(buffer_size)):
            dataset_index = index_order[i]
            image_start = dataset_index
            image_end = image_start + context_length
            image_tensor = torch.FloatTensor(self.context_length, 3, self.resize.size[0], self.resize.size[1])
            tensor_idx = 0
            for image_index in range(image_start, image_end):
                fp, ts, steering, throttle, brake = self.annotations[image_index].split(",")
                impil = PILImage.open( os.path.join( self.image_directory, fp ) )
                #print(impil)
                im = self.totensor( self.resize( impil ) )
                image_tensor[tensor_idx] = im
                tensor_idx += 1
            self.image_dict[dataset_index] = image_tensor
            self.image_leader_index += 1

            label_start = image_end
            label_end = label_start + self.sequence_length
            tensor_idx = 0
            label_tensor = torch.FloatTensor(self.sequence_length, 3)
            for label_index in range(label_start, label_end):
                _, ts, steering, throttle, brake = self.annotations[label_index].split(",")
                label_tensor[tensor_idx][0] = float(steering)
                label_tensor[tensor_idx][1] = float(throttle)
                label_tensor[tensor_idx][2] = float(brake)
                tensor_idx += 1
            self.label_dict[dataset_index] = label_tensor
            self.label_leader_index += 1
    def getLabelRange(self, index : int):
        while index not in self.label_dict:
            time.sleep(0.1)
        self.label_lock.acquire()
        labels = self.label_dict.pop(index)
        self.label_lock.release()
        return labels
    def numberOfImages(self):
        return len(self.annotations)
    def getImageRange(self, index : int):
        while index not in self.image_dict:
            time.sleep(0.1)
        self.image_lock.acquire()
        images = self.image_dict.pop(index)
        self.image_lock.release()
        return images
    def imageWorker(self):
        pass
    def labelWorker(self):
        pass
class DeepF1ImageTensorBackend(DeepF1ImageSequenceBackend):
    def __init__(self, context_length : int, sequence_length : int, image_tensor : torch.Tensor = None, label_tensor : torch.Tensor = None):
        super(DeepF1ImageTensorBackend, self).__init__(context_length, sequence_length)
        self.image_tensor : torch.Tensor = image_tensor
        self.label_tensor : torch.Tensor = label_tensor
    
    def numberOfImages(self):
        return self.image_tensor.shape[0]

    def getImageRange(self, index: int):
        images_start = index
        images_end = images_start + self.context_length
               
        labels_start = images_end
        labels_end = labels_start + self.sequence_length
        return self.image_tensor[images_start:images_end]


    def getLabelRange(self, index: int):
        images_start = index
        images_end = images_start + self.context_length
        labels_start = images_end
        labels_end = labels_start + self.sequence_length
        return self.image_tensor[labels_start:labels_end]


    def loadImages(self, annotation_file, im_size):
        f = open(annotation_file)
        annotations = f.readlines()
        f.close()
        num_lines = len(annotations)
        image_folder = os.path.join(os.path.dirname(annotation_file),"raw_images")
        resize = torchvision.transforms.Resize(im_size)
        totensor = torchvision.transforms.ToTensor()
        self.image_tensor = torch.Tensor(num_lines, 3, im_size[0], im_size[1])
        self.image_tensor.type(torch.float32)
        self.label_tensor = torch.Tensor(num_lines, 3)
        self.label_tensor.type(torch.float32)
        
        for idx in tqdm(range(len(annotations)),desc='Loading Data',leave=True):
            fp, ts, steering, throttle, brake = annotations[idx].split(",")
            impil = PILImage.open( os.path.join( image_folder, fp ) )
            #print(impil)
            im = totensor( resize( impil ) )
            self.image_tensor[idx] = im

            self.label_tensor[idx][0] = float(steering)
            self.label_tensor[idx][1] = float(throttle)
            self.label_tensor[idx][2] = float(brake)
