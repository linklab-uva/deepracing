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
    def numberOfImages(self):
        raise NotImplementedError("DeepF1ImageBackend classes must implement numberOfImages(self)")
class DeepF1ImageDirectoryBackend(DeepF1ImageBackend):
    def __init__(self, annotation_file, im_size = (66,200)):
        super(DeepF1ImageDirectoryBackend, self).__init__()
        f = open(annotation_file)
        self.annotations = f.readlines()
        f.close()        
        self.resize = torchvision.transforms.Resize(im_size)
        self.totensor = torchvision.transforms.ToTensor()
        self.images_dir = os.path.join(os.path.dirname(annotation_file),"raw_images") 
    
    def numberOfImages(self):
        return len(self.annotations)

    def getImage(self, index):
        fp, ts, steering, throttle, brake = self.annotations[index].split(",")
        impil = PILImage.open( os.path.join( self.images_dir, fp ) )
        #print(impil)
        im = self.totensor( self.resize( impil ) )
        return im


    def getImageRange(self, start, end):
        images = torch.FloatTensor(end - start, 3, self.resize.size[0], self.resize.size[1])
        for index in range(images.shape[0]):
            images[index] = self.getImage(index)
        return images


    def getLabel(self, index):
        fp, ts, steering, throttle, brake = self.annotations[index].split(",")
        label=torch.FloatTensor(3)
        label[0] = float(steering)
        label[1] = float(throttle)
        label[2] = float(brake)
        return label


    def getLabelRange(self, start, end):
        labels = torch.FloatTensor(end - start, 3)
        for index in range(labels.shape[0]):
            labels[index] = self.getLabel(index)
        return labels


class DeepF1ImageTensorBackend(DeepF1ImageBackend):
    def __init__(self, image_tensor = None, label_tensor= None):
        super(DeepF1ImageTensorBackend, self).__init__()
        self.image_tensor = image_tensor
        self.label_tensor = label_tensor
    
    def numberOfImages(self):
        return self.image_tensor.shape[0]

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

    
    
