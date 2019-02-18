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
import random
from multiprocessing import Process
from multiprocessing import Lock as MPLock
import cv2
import deepf1_image_reading as imreading
import lmdb
class DeepF1ImageSequenceBackend(metaclass=abc.ABCMeta):
    @abc.abstractmethod
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
    def __indexRanges__(self, dataset_index : int):
        image_start = dataset_index
        image_end = image_start + self.context_length
        label_start = image_end
        label_end = label_start + self.sequence_length
        return image_start, image_end, label_start, label_end
class DeepF1LMDBBackend(DeepF1ImageSequenceBackend):
    def __init__(self, annotation_file : str, context_length : int, sequence_length : int, imsize=(66,200)):
        super(DeepF1LMDBBackend, self).__init__(context_length, sequence_length)
        f = open(annotation_file, 'r')
        annotations = f.readlines()
        f.close()
        self.image_files : list = []
        self.labels = np.empty((len(annotations), 3), dtype=np.float32)
        self.image_directory = os.path.join(os.path.dirname(annotation_file),'raw_images')
        self.image_db = None
        self.txn = None
        self.env = None#lmdb.open(os.path.join(lmdb_dir,data_file+'.mdb'))
        print("Loading image labels and file paths")
        for (i, line) in tqdm(enumerate(annotations)):
            fp, _, steering, throttle, brake = line.replace("\n","").split(",")
            self.image_files.append(fp)
            self.labels[i][0] = float(steering)
            self.labels[i][1] = float(throttle)
            self.labels[i][2] = float(brake)
        self.totensor=torchvision.transforms.ToTensor()
        self.resize=torchvision.transforms.Resize(imsize)
    def writeDatabase(self, db_path : str):
        if not os.path.isdir(db_path):
            os.makedirs(db_path)
        env = lmdb.open(db_path, map_size=1e9)
        with env.begin(write=True) as write_txn:
            print("Loading image data")
            for (i,fp) in tqdm(enumerate(self.image_files)):
                impil = PILImage.open(os.path.join(self.image_directory, fp))
                imnp = np.array(self.resize(impil)).astype(np.uint8).transpose(2,0,1)
                write_txn.put(fp.encode('ascii'),imnp.flatten().tostring())
        self.env = lmdb.open(db_path, map_size=1e6)
    def readDatabase(self, db_path : str):
        if not os.path.isdir(db_path):
            raise IOError("Path " + db_path + " is not a directory")
        self.env = lmdb.open(db_path, map_size=1e6)
    def getImageRange(self, index : int):
        image_start, image_end, _, _ = self.__indexRanges__(index)
        size = (image_end - image_start, 3, self.resize.size[0], self.resize.size[1] )
        images = np.empty(size, dtype=np.uint8)
        with self.env.begin(write=False) as read_txn:
            for (i,fp) in enumerate(self.image_files[image_start:image_end]):
                im_flat = read_txn.get(fp.encode('ascii'))
                images[i] = np.frombuffer( im_flat,dtype=np.uint8).reshape(size[1:])
        return images.astype(np.float32)/255.0
    def getLabelRange(self, index : int):
        _, _, label_start, label_end = self.__indexRanges__(index)
        labels = self.labels[label_start:label_end]
        return labels
    def numberOfImages(self):
        return len(self.image_files)
class DeepF1ImageDirectoryBackend(DeepF1ImageSequenceBackend):
    def __init__(self, annotation_file : str, context_length : int, sequence_length : int, imsize=(66,200)):
        super(DeepF1ImageDirectoryBackend, self).__init__(context_length, sequence_length)
        f = open(annotation_file, 'r')
        annotations = f.readlines()
        f.close()
        self.image_file_map : dict = {}
        self.labels = np.empty((len(annotations), 3))
        for (i, line)in enumerate(annotations):
            fp, ts, steering, throttle, brake = line.split(",")
            self.image_file_map[i] = fp
            self.labels[i][0] = float(steering)
            self.labels[i][1] = float(throttle)
            self.labels[i][2] = float(brake)

        self.totensor=torchvision.transforms.ToTensor()
        self.resize=torchvision.transforms.Resize(imsize)
        self.grayscale=torchvision.transforms.Grayscale()
        self.image_directory = os.path.join(os.path.dirname(annotation_file),'raw_images')

    def getImageRange(self, index : int):
        image_start, image_end, _, _ = self.__indexRanges__(index)
        l = imreading.readImages(os.path.join(self.image_directory,'raw_image_'), image_start, image_end - image_start, cv2.IMREAD_GRAYSCALE, self.resize.size )
        im_array = np.array( l ) / 255.0
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

    def numberOfImages(self):
        return self.labels.shape[0]

class DeepF1LeaderFollowerBackend(DeepF1ImageSequenceBackend):
    def __init__(self, annotation_file : str, context_length : int, sequence_length : int, buffer_size : int):
        super(DeepF1LeaderFollowerBackend, self).__init__(context_length, sequence_length)
        self.annotation_file : str = annotation_file
        self.image_directory = os.path.join(os.path.dirname(annotation_file),'raw_images')
        f = open(self.annotation_file, 'r')
        self.annotations : List[str] = f.readlines()
        f.close()
        self.image_dict : dict = {}
        self.image_lock = MPLock()
        self.image_leader_index : int = 0
        self.label_dict : dict = {}
        self.label_lock = MPLock()
        self.label_leader_index : int = 0

        __l = range(len(self.annotations) - context_length - sequence_length )
        
        self.index_order : np.array =  np.array( __l, order = 'C' )
        np.random.shuffle(self.index_order)
        self.buffer_size = buffer_size
        print('Prefilling the buffer')
        self.resize = torchvision.transforms.Resize((66,200))
        self.totensor = torchvision.transforms.ToTensor()
        self.running=True
        for i in tqdm(range(buffer_size)):
            self.image_lock.acquire()
            dataset_index = self.index_order[self.image_leader_index]
            self.image_leader_index += 1
            self.image_lock.release()
            self.__loadImages__(dataset_index)

            
            self.label_lock.acquire()
            dataset_index = self.index_order[self.label_leader_index]
            self.label_leader_index += 1
            self.label_lock.release()
            self.__loadLabels__(dataset_index)
        p1 = Process(target=self.imageWorker).start()
        p2 = Process(target=self.labelWorker).start()

    

    def __loadImages__(self, dataset_index):
        image_start, image_end, _, _ = self.__indexRanges__(dataset_index)

        image_tensor = torch.FloatTensor(self.context_length, 3, self.resize.size[0], self.resize.size[1])
        tensor_idx = 0
        for image_index in range(image_start, image_end):
            fp, _, _, _, _ = self.annotations[image_index].split(",")
            impil = PILImage.open( os.path.join( self.image_directory, fp ) )
            #print(impil)
            im = self.totensor( self.resize( impil ) )
            image_tensor[tensor_idx] = im
            tensor_idx += 1
        self.image_lock.acquire()
        self.image_dict[dataset_index] = image_tensor
        self.image_lock.release()
    def __loadLabels__(self, dataset_index):
        _, _, label_start, label_end = self.__indexRanges__(dataset_index)
        tensor_idx = 0
        label_tensor = torch.FloatTensor(self.sequence_length, 3)
        for label_index in range(label_start, label_end):
            _, ts, steering, throttle, brake = self.annotations[label_index].split(",")
            label_tensor[tensor_idx][0] = float(steering)
            label_tensor[tensor_idx][1] = float(throttle)
            label_tensor[tensor_idx][2] = float(brake)
            tensor_idx += 1
        self.label_lock.acquire()
        self.label_dict[dataset_index] = label_tensor
        self.label_lock.release()

    def getLabelRange(self, index : int, pop : bool = True):
        while index not in self.label_dict:
            print('Waiting for label at index: %d' %(index))
            time.sleep(0.05)
        if pop:
            self.label_lock.acquire()
            labels = self.label_dict.pop(index)
            self.label_lock.release()
        else:
            labels = self.label_dict[index]
        return labels
    def getImageRange(self, index : int, pop : bool = True):
        while index not in self.image_dict:
            print('Waiting for image at index: %d' %(index))
            time.sleep(0.05)
        if pop:
            self.image_lock.acquire()
            images = self.image_dict.pop(index)
            self.image_lock.release()
        else:
            images = self.image_dict[index]
        return images
    def numberOfImages(self):
        return len(self.annotations)
    def imageWorker(self):
        print('Spawned an image worker thread')
        while self.running and self.image_leader_index < len(self.index_order):
            if True or (len(self.image_dict) < self.buffer_size):
              #  print('Waiting for image lock')
                self.image_lock.acquire()
                dataset_index = self.index_order[self.image_leader_index]
                print('Loading image at index: %d' %(dataset_index))
                self.image_leader_index += 1
                self.image_lock.release()
                self.__loadImages__(dataset_index)
        print('Exited an image worker thread')
    def labelWorker(self):
        print('Spawned a label worker thread')
        while self.running and self.label_leader_index < len(self.index_order):
            if True or (len(self.label_dict) < self.buffer_size):
             #   print('Waiting for label lock')
                self.label_lock.acquire()
                dataset_index = self.index_order[self.label_leader_index]
              #  print('Loading label at index: %d' %(dataset_index))
                self.label_leader_index += 1
                self.label_lock.release()
                self.__loadLabels__(dataset_index)
        print('Exited a label worker thread')
    def stop(self):
        self.running=False
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
