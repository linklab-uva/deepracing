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
import torchvision
import PIL
from PIL import Image as PILImage
import data_loading.backend.ImageSequenceBackend as image_backends
import data_loading.backend.OpticalFlowBackend as of_backends
from torch.utils.data.sampler import Sampler as TorchSampler
def imagesToFlow(images, totensor = torchvision.transforms.ToTensor()):
    flows=torch.zeros((images.shape[0]-1,2,images.shape[2],images.shape[3]), dtype=torch.float32)
    im = images[0]
    im_np = im.numpy().transpose(1,2,0)
    first = (255*im_np).astype(np.uint8)
    first_gray = cv2.cvtColor(first, cv2.COLOR_RGB2GRAY)
  #  first_bgr = cv2.cvtColor(first, cv2.COLOR_RGB2BGR)
    for idx in range(1, images.shape[0]):
        im = images[idx]
        im_np = im.numpy().transpose(1,2,0)
        second = (255*im_np).astype(np.uint8)
        second_gray = cv2.cvtColor(second, cv2.COLOR_RGB2GRAY)
        #second_bgr = cv2.cvtColor(second, cv2.COLOR_RGB2BGR)

        flow = cv2.calcOpticalFlowFarneback(first_gray,second_gray, None, 0.5, 3, 20, 8, 5, 1.2, 0).astype(np.float32)
        flows[idx-1] = totensor(flow)
        first_gray = second_gray
    return flows
class DeterministicSampler(TorchSampler):
    def __init__(self, indices):
        self.indices = indices
    def __iter__(self):
        return iter(self.indices)
    def __len__(self):
        return len(self.indices)
class F1ImageSequenceDataset(Dataset):
    def __init__(self, backend : image_backends.DeepF1ImageSequenceBackend):
        super(F1ImageSequenceDataset, self).__init__()
        self.backend : image_backends.DeepF1ImageSequenceBackend=backend
        self.len = self.backend.numberOfImages() - backend.context_length - backend.sequence_length
    
    def __getitem__(self, index):
        images = self.backend.getImageRange(index)
        labels = self.backend.getLabelRange(index)
        return torch.Tensor(images.astype(np.float32)/255.0) , torch.Tensor(labels)
    def __len__(self):
        return self.len
class F1OpticalFlowDataset(Dataset):
    def __init__(self, backend : of_backends.DeepF1OptFlowBackend):
        super(F1OpticalFlowDataset, self).__init__()
        self.backend : of_backends.DeepF1OptFlowBackend = backend
        self.len = self.backend.numberOfFlowImages() - backend.context_length - backend.sequence_length
    def __getitem__(self, index):
        images, flows = self.backend.getFlowImageRange(index)
        cat = np.concatenate((images[1:].astype(np.float32)/255.0,flows), axis=1)
        labels = self.backend.getLabelRange(index)
        return torch.Tensor(cat), torch.Tensor(labels)
    def __len__(self):
        return self.len

class F1CombinedDataset(Dataset):
    def __init__(self, annotation_filepath, im_size, context_length = 25, sequence_length=25):
        super(F1CombinedDataset, self).__init__()
        self.context_length=context_length
        self.sequence_length=sequence_length
        self.totensor = torchvision.transforms.ToTensor()
        self.grayscale = torchvision.transforms.Grayscale()
        self.resize = torchvision.transforms.Resize(im_size)
        self.annotation_filename = os.path.basename(annotation_filepath)
        self.annotations = open(annotation_filepath).readlines()
        self.root_folder = os.path.dirname(annotation_filepath)
        self.image_folder = os.path.join(self.root_folder,'raw_images')
        self.len = len(self.annotations) - context_length - sequence_length - 1
        self.images = None
        self.labels = None
        self.image_pickle_postfix = "_combined_images.pt"
        self.label_pickle_postfix = "_combined_labels.pt"
    def loadPickles(self):
        
        splits = self.annotation_filename.split(".")
        prefix=splits[0]

        lblname = prefix + self.label_pickle_postfix
        imgname = prefix + self.image_pickle_postfix


        self.images = torch.load(os.path.join(self.root_folder,imgname))


        self.labels = torch.load(os.path.join(self.root_folder,lblname))

    def writePickles(self):
        splits = self.annotation_filename.split(".")
        prefix=splits[0]
        lblname = prefix + self.label_pickle_postfix
        imgname = prefix + self.image_pickle_postfix
        torch.save(self.labels,open(os.path.join(self.root_folder,lblname), 'w+b'))
        torch.save(self.images,open(os.path.join(self.root_folder,imgname), 'w+b'))
    def loadFiles(self):
        fp, ts, steering, throttle, brake = self.annotations[0].split(",")
        im = self.totensor( self.grayscale( self.resize( PILImage.open( os.path.join( self.image_folder, fp ) ) ) ) ).type(torch.float32)
        prvs_img =  np.round( 255.0 * im[0].numpy() ).astype(np.uint8)

        self.images = torch.zeros(len(self.annotations) - 1, 3, self.resize.size[0], self.resize.size[1], dtype = torch.float32)
        self.labels = torch.zeros(len(self.annotations) - 1, 3, dtype = torch.float32)
        for idx in tqdm(range(1,len(self.annotations)),desc='Loading Data',leave=True):
            fp, ts, steering, throttle, brake = self.annotations[idx].split(",")
            im = self.totensor( self.grayscale( self.resize( PILImage.open( os.path.join( self.image_folder, fp ) ) ) ) ).type(torch.float32) 
          #  print(im.size())
            next_img = np.round( 255.0 * im[0].numpy() ).astype(np.uint8)

            flow = cv2.calcOpticalFlowFarneback(prvs_img,next_img, None, 0.5, 3, 20, 8, 5, 1.2, 0).astype(np.float32)
            indx = idx-1
            self.images[indx][0:2] = self.totensor(flow)
            self.images[indx][2] = im[0]

            self.labels[indx][0] = float(steering)
            self.labels[indx][1] = float(throttle)
            self.labels[indx][2] = float(brake)
            prvs_img = next_img
    def __getitem__(self, index):
        if(self.images is not None and self.labels is not None):
            images_start = index
            images_end = images_start + self.context_length - 1
            images = self.images[ index : images_end + 1]
                    
            labels_start = images_end
            labels_end = labels_start + self.sequence_length
            labels = self.labels[labels_start : labels_end]
        else:
            raise( NotImplementedError("Only preloading images is supported for OpticalFlow dataset") )
        return images , labels
    def __len__(self):
        return self.len
class F1ImageDataset(Dataset):
    def __init__(self, annotation_filepath, im_size):
        super(F1ImageDataset, self).__init__()

        self.totensor = torchvision.transforms.ToTensor()
        self.grayscale = torchvision.transforms.Grayscale()
        self.resize = torchvision.transforms.Resize(im_size)
        self.annotation_filename = os.path.basename(annotation_filepath)
        self.annotations = open(annotation_filepath).readlines()
        self.root_folder = os.path.dirname(annotation_filepath)
        self.image_folder = os.path.join(self.root_folder,'raw_images')
        self.len = len(self.annotations)
        self.images = None
        self.labels = None
        
        self.image_pickle_postfix = "_static_images.pt"
        self.label_pickle_postfix = "_static_labels.pt"
    def loadPickles(self):
        
        splits = self.annotation_filename.split(".")
        prefix=splits[0]

        lblname = prefix + self.label_pickle_postfix
        imgname = prefix + self.image_pickle_postfix


        self.images = torch.load(os.path.join(self.root_folder,imgname))


        self.labels = torch.load(os.path.join(self.root_folder,lblname))

    def writePickles(self):
        splits = self.annotation_filename.split(".")
        prefix=splits[0]
        lblname = prefix + self.label_pickle_postfix
        imgname = prefix + self.image_pickle_postfix
        torch.save(self.labels,open(os.path.join(self.root_folder,lblname), 'w+b'))
        torch.save(self.images,open(os.path.join(self.root_folder,imgname), 'w+b'))
    def loadFiles(self):
        window_name = "image"
        # cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        self.images = torch.zeros(len(self.annotations), 3, self.resize.size[0], self.resize.size[1], dtype = torch.float32)
        self.labels = torch.zeros(len(self.annotations), 3, dtype = torch.float32)
        for idx in tqdm(range(len(self.annotations)),desc='Loading Data',leave=True):
            fp, ts, steering, throttle, brake = self.annotations[idx].split(",")
            im = self.totensor( self.resize( PILImage.open( os.path.join( self.image_folder, fp ) ) ) )

            
            # im_np = cv2.cvtColor(im.numpy().transpose(1,2,0), cv2.COLOR_RGB2BGR)
            # cv2.imshow(window_name, im_np)
            # cv2.waitKey(1)

            self.images[idx] = im
            self.labels[idx][0] = float(steering)
            self.labels[idx][1] = float(throttle)
            self.labels[idx][2] = float(brake)
    def __getitem__(self, index):
        #print("Getting an image at index: %d" % (index))
        if(self.images is not None and self.labels is not None):
            image = self.images[index]
            label = self.labels[index]
        else:
            fp, ts, steering, throttle, brake = self.annotations[index].split(",")
            image = self.totensor( self.resize( PILImage.open( os.path.join( self.image_folder, fp ) ) ) )
            label = torch.zeros(3)
            label[0] = float(steering)
            label[1] = float(throttle)
            label[2] = float(brake)
        return image , label
    def __len__(self):
        return self.len