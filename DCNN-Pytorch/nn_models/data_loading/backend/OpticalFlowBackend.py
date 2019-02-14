import torch
import os
import PIL
from PIL import Image as PILImage
import torchvision.transforms as transforms
import torchvision
from tqdm import tqdm as tqdm
import cv2
import numpy as np
class DeepF1OpticalFlowBackend():
    def __init__(self):
        pass
    def getFlowImage(self, index):
        raise NotImplementedError("DeepF1OpticalFlowBackend classes must implement getFlowImage(self, index)")
    def getFlowImageRange(self, start, end):
        raise NotImplementedError("DeepF1OpticalFlowBackend classes must implement getFlowImageRange(self, start, end)")
    def getLabel(self, index):
        raise NotImplementedError("DeepF1OpticalFlowBackend classes must implement getLabel(self, index)")
    def getLabelRange(self, start, end):
        raise NotImplementedError("DeepF1OpticalFlowBackend classes must implement getLabelRange(self, start, end)")
    def numberOfFlowImages(self):
        raise NotImplementedError("DeepF1OpticalFlowBackend classes must implement numberOfFlowImages(self)")
class DeepF1OpticalFlowTensorBackend(DeepF1OpticalFlowBackend):
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

    
    

