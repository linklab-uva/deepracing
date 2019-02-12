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
    def __init__(self, flow_tensor = None, label_tensor= None):
        super(DeepF1OpticalFlowTensorBackend, self).__init__()
        self.flow_tensor = flow_tensor
        self.label_tensor = label_tensor
    
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


    def loadFlows(self, annotation_file, im_size= (66,200)):
        f = open(annotation_file)
        annotations = f.readlines()
        f.close()
        num_lines = len(annotations)
        image_folder = os.path.join(os.path.dirname(annotation_file),"raw_images")
        resize = torchvision.transforms.Resize(im_size)
        totensor = torchvision.transforms.ToTensor()
        self.flow_tensor = torch.FloatTensor(num_lines-1, 2, im_size[0], im_size[1])
        self.label_tensor = torch.FloatTensor(num_lines-1, 3)
        
        fp, ts, steering, throttle, brake = annotations[0].split(",")
        impil = PILImage.open( os.path.join( image_folder, fp ) )
        im = totensor( resize( impil ) )
        im_np = im.numpy().transpose(1,2,0)
        first = (255*im_np).astype(np.uint8)
        first_gray = cv2.cvtColor(first, cv2.COLOR_RGB2GRAY)
        first_bgr = cv2.cvtColor(first, cv2.COLOR_RGB2BGR)
        for idx in tqdm(range(1, len(annotations)),desc='Loading Data',leave=True):
            fp, ts, steering, throttle, brake = annotations[idx].split(",")
            impil = PILImage.open( os.path.join( image_folder, fp ) )
            #print(impil)
            im = totensor( resize( impil ) )
            im_np = im.numpy().transpose(1,2,0)
            second = (255*im_np).astype(np.uint8)
            second_gray = cv2.cvtColor(second, cv2.COLOR_RGB2GRAY)
            second_bgr = cv2.cvtColor(second, cv2.COLOR_RGB2BGR)

            flow = cv2.calcOpticalFlowFarneback(first_gray,second_gray, None, 0.5, 3, 20, 8, 5, 1.2, 0).astype(np.float32)
            self.flow_tensor[idx-1] = totensor(flow)
            self.label_tensor[idx-1][0] = float(steering)
            self.label_tensor[idx-1][1] = float(throttle)
            self.label_tensor[idx-1][2] = float(brake)

            first_gray = second_gray

    
    

