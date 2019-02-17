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
class DeepF1OptFlowDirectoryBackend(DeepF1OptFlowBackend):
    def __init__(self, annotation_file : str, context_length : int, sequence_length : int, imsize=(66,200)):
        super(DeepF1OptFlowDirectoryBackend, self).__init__(context_length, sequence_length)
        f = open(annotation_file, 'r')
        annotations = f.readlines()
        f.close()
        self.labels = np.empty((len(annotations) - 1, 3))
        for (i, line)in enumerate(annotations[1:]):
            _, _, steering, throttle, brake = line.split(",")
            self.labels[i][0] = float(steering)
            self.labels[i][1] = float(throttle)
            self.labels[i][2] = float(brake)

        self.totensor=torchvision.transforms.ToTensor()
        self.resize=torchvision.transforms.Resize(imsize)
        self.grayscale=torchvision.transforms.Grayscale()
        self.image_directory = os.path.join(os.path.dirname(annotation_file),'raw_images')

    def getFlowImageRange(self, index : int):
        image_start, image_end, _, _ = self.__indexRanges__(index)
        l = imreading.readImageFlows(os.path.join(self.image_directory,'raw_image_'), image_start, image_end - image_start, self.resize.size )
        im_array = np.array( l ).transpose(0,3,1,2)
        # im_array = np.empty((image_end - image_start, 3, self.resize.size[0], self.resize.size[1]))
        # for (i, index) in enumerate(range(image_start, image_end)):
        #     fp = os.path.join(self.image_directory,'raw_image_'+str(index+1)+'.jpg')
        #     image = imreading.readImage( os.path.join( self.image_directory ,fp ) ,cv2.IMREAD_GRAYSCALE )
        #     imresize = cv2.resize( image, ( self.resize.size[1], self.resize.size[0] ) ) 
        #     im_array[i] = (imresize.astype(np.float32)/255.0)
        return im_array
    def getLabelRange(self, index : int):
        _, _, label_start, label_end = self.__indexRanges__(index)
        # label_array = np.empty((label_end - label_start, 3), dtype = np.float32) 
        # #array_idx = 0
        # for (i,line) in enumerate(self.annotations[label_start: label_end]):
        #     _, ts, steering, throttle, brake = line.split(",")
        #     label_array[i][0] = float(steering)
        #     label_array[i][1] = float(throttle)
        #     label_array[i][2] = float(brake)
        #     #array_idx += 1
        return self.labels[label_start:label_end]

    def numberOfFlowImages(self):
        return self.labels.shape[0]

class DeepF1OpticalFlowTensorBackend(DeepF1OptFlowBackend):
    def __init__(self):
        super(DeepF1OpticalFlowTensorBackend, self).__init__()
        self.flow_tensor = None
        self.label_tensor = None
        self.flow_source_file = None
        self.label_source_file = None

    def numberOfFlowImages(self):
        return self.flow_tensor.shape[0]

    def getFlowImage(self, index):
        return self.flow_tensor[index]

    def getFlowImageRange(self, start, end):
        return self.flow_tensor[start:end]

    def getLabel(self, index):
        return self.label_tensor[index]

    def getLabelRange(self, start, end):
        return self.label_tensor[start:end]

    def loadPickles(self, flow_fp, label_fp):
        del self.flow_tensor
        del self.label_tensor
        self.flow_source_file = flow_fp
        self.label_source_file = label_fp
        self.flow_tensor = torch.load(self.flow_source_file)
        self.label_tensor = torch.load(self.label_source_file)

    def loadFlows(self, annotation_file, im_size=(66,200)):
        f = open(annotation_file)
        annotations = f.readlines()
        f.close()
        num_lines = len(annotations)
        image_folder = os.path.join(os.path.dirname(annotation_file),"raw_images")
        resize = torchvision.transforms.Resize(im_size)
        totensor = torchvision.transforms.ToTensor()
        greyscale = torchvision.transforms.Grayscale()
        self.flow_tensor = torch.FloatTensor(num_lines-1, 2, im_size[0], im_size[1])
        self.label_tensor = torch.FloatTensor(num_lines-1, 3)
        
        fp, ts, steering, throttle, brake = annotations[0].split(",")
        impil = PILImage.open( os.path.join( image_folder, fp ) )
        first = np.array(greyscale( resize( impil ) ) )
        for idx in tqdm(range(1, len(annotations)),desc='Loading Data',leave=True):
            fp, ts, steering, throttle, brake = annotations[idx].split(",")
            impil = PILImage.open( os.path.join( image_folder, fp ) )

            second = np.array(greyscale( resize( impil ) ) )
            flow = cv2.calcOpticalFlowFarneback(first,second, None, 0.5, 3, 20, 8, 5, 1.2, 0).astype(np.float32)
            self.flow_tensor[idx-1] = totensor(flow)
            self.label_tensor[idx-1][0] = float(steering)
            self.label_tensor[idx-1][1] = float(throttle)
            self.label_tensor[idx-1][2] = float(brake)

            first = second

    
    

