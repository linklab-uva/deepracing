import torch
import os
import PIL
from PIL import Image as PILImage
import torchvision.transforms as transforms
import torchvision
from tqdm import tqdm as tqdm
class DeepF1ImageBackend():
    def __init__(self):
        pass
    def getImage(self, index):
        raise NotImplementedError("DeepF1ImageBackend classes must implement getImage(self, index)")
    def getImageRange(self, start, end):
        raise NotImplementedError("DeepF1ImageBackend classes must implement getImageRange(self, start, end)")
    def getLabel(self, index):
        raise NotImplementedError("DeepF1ImageBackend classes must implement getLabel(self, index)")
    def getLabelRange(self, start, end):
        raise NotImplementedError("DeepF1ImageBackend classes must implement getLabelRange(self, start, end)")


class DeepF1ImageTensorBackend(DeepF1ImageBackend):
    def __init__(self, image_tensor = None, label_tensor= None):
        super(DeepF1ImageTensorBackend, self).__init__()
        self.image_tensor = image_tensor
        self.label_tensor = label_tensor


    def getImage(self, index):
        return self.image_tensor[index]


    def getImageRange(self, start, end):
        return self.image_tensor[start:end]


    def getLabel(self, index):
        return self.label_tensor[index]


    def getLabelRange(self, start, end):
        return self.label_tensor[start:end]


    def loadImages(self, annotation_file, im_size):
        f = open(annotation_file)
        annotations = f.readlines()
        f.close()
        num_lines = len(annotations)
        image_folder = os.path.join(os.path.dirname(annotation_file),"raw_images")
        resize = torchvision.transforms.Resize(im_size)
        totensor = torchvision.transforms.ToTensor()
        self.image_tensor = torch.Tensor(num_lines, 3, 66, 200)
        self.image_tensor.type(dtype='torch.float32')
        self.label_tensor = torch.Tensor(num_lines, 3)
        self.label_tensor.type(dtype='torch.float32')
        
        for idx in tqdm(range(len(annotations)),desc='Loading Data',leave=True):
            fp, ts, steering, throttle, brake = annotations[idx].split(",")
            impil = PILImage.open( os.path.join( image_folder, fp ) )
            im = totensor( resize( impil ) )
            self.image_tensor[idx] = im

            self.label_tensor[idx][0] = float(steering)
            self.label_tensor[idx][1] = float(throttle)
            self.label_tensor[idx][2] = float(brake)

    
    
