import torch
import os
import PIL
from PIL import Image as PILImage
import torchvision.transforms as transforms
import torchvision
from tqdm import tqdm as tqdm
import cv2
import numpy as np
import abc
import deepf1_image_reading as imreading
import data_loading.backend.ImageSequenceBackend as image_backends
import lmdb
class DeepF1OptFlowBackend(metaclass=abc.ABCMeta):
    def __init__(self, context_length : int, sequence_length : int):
        self.context_length = context_length
        self.sequence_length = sequence_length

    @abc.abstractmethod
    def getFlowImageRange(self, index : int):
        pass
    
    @abc.abstractmethod
    def getLabelRange(self, index : int):
        pass

    @abc.abstractmethod
    def numberOfFlowImages(self):
        pass
    def __indexRanges__(self, dataset_index : int):
        image_start = dataset_index
        image_end = image_start + self.context_length
        label_start = image_end
        label_end = label_start + self.sequence_length
        return image_start, image_end, label_start, label_end
def npimagesToFlow(images : np.ndarray):
    images_transposed = images.transpose(0,2,3,1)
    grayscale_images = np.empty((images.shape[0], 1, images.shape[2], images.shape[3]), dtype=np.uint8)
    grayscale_images[0][0] = cv2.cvtColor(images_transposed[0], cv2.COLOR_RGB2GRAY)
    flows = np.empty((images.shape[0]-1, images.shape[2], images.shape[3], 2), dtype=np.float32)
    for idx in range(1, images.shape[0]):
        grayscale_images[idx][0] = cv2.cvtColor(images_transposed[idx], cv2.COLOR_RGB2GRAY)
        flows[idx-1] = cv2.calcOpticalFlowFarneback(grayscale_images[idx-1][0], grayscale_images[idx][0], None, 0.5, 3, 20, 8, 5, 1.2, 0).astype(np.float32)
    return grayscale_images, flows.transpose(0,3,1,2)
class DeepF1LMDBOptFlowBackend(DeepF1OptFlowBackend):
    def __init__(self, annotation_file : str, context_length : int, sequence_length : int, imsize=(66,200)):
        super(DeepF1LMDBOptFlowBackend, self).__init__(context_length, sequence_length)
        self.image_backend : image_backends.DeepF1LMDBBackend = image_backends.DeepF1LMDBBackend(annotation_file, context_length+1, sequence_length, imsize=imsize)
        
    def getFlowImageRange(self, index : int):
        images = self.image_backend.getImageRange(index)
        return npimagesToFlow(images)
    def getLabelRange(self, index : int):
        return self.image_backend.getLabelRange(index)
    def numberOfFlowImages(self):
        return self.image_backend.numberOfImages() - 1
    def readImages(self, db_path : str):
        self.image_backend.readImages(db_path)
    def readDatabase(self, db_path : str):
        self.image_backend.readDatabase(db_path)