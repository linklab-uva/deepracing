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
    def __init__(self, root_folder, annotation_filepath, im_size, use_float32=False, img_transformation = None, label_transformation = None, optical_flow=False):
        super(F1Dataset, self).__init__()
        self.im_size=im_size
        self.label_size = 1
        self.use_float32=use_float32
        self.root_folder = root_folder
        self.img_transformation = img_transformation
        self.label_transformation = label_transformation
        self.annotation_filepath = annotation_filepath
        self.optical_flow=optical_flow
        self.annotations_file = open(os.path.join(self.root_folder,self.annotation_filepath), "r")
        self.annotations = self.annotations_file.readlines()
        self.partition_size=min(80000,len(self.annotations))
        if optical_flow:
            self.length = self.partition_size- 1
            self.images = np.tile(0, (self.length,2,im_size[0],im_size[1])).astype(np.float32)
        else:
            self.length = self.partition_size
            self.images = np.tile(0, (self.length,3,im_size[0],im_size[1])).astype(np.int8)
        self.labels = np.tile(0, (self.length)).astype(np.float64)
        self.throttle = np.tile(0, (self.length)).astype(np.float64)
        self.brake = np.tile(0, (self.length)).astype(np.float64)
        self.preloaded=False
    def statistics(self):
        #print('\t\t-->Averaging array with shape: ', self.images.shape)
        mean = np.mean(self.images,(0,2,3))
        stdev = np.std(self.images,(0,2,3))
        return mean,stdev
    def write_pickles(self,image_pickle, label_pickle):
        filename = image_pickle
        fp = open(filename, 'wb')
        pickle.dump(self.images, fp, protocol=4)
        fp.close()
        print('File %s is saved.' % filename)

        filename = label_pickle
        fp = open(filename, 'wb')
        pickle.dump(self.labels, fp, protocol=4)
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
    def read_files_flow(self):
        pickle_dir,_ = self.annotation_filepath.split('.')
        pickle_dir+='_data'
        if(not os.path.exists(pickle_dir)):
            os.makedirs(pickle_dir)
        print("loading data and computing optical flow")
        fp, ts, steering, throttle, brake = self.annotations[0].split(",")
        prvs = load_image(os.path.join(self.root_folder,"raw_images",fp)).astype(np.float32) / 255.0
        prvs_grayscale = cv2.cvtColor(prvs,cv2.COLOR_BGR2GRAY)
        prvs_resize = cv2.resize(prvs_grayscale, (self.im_size[1], self.im_size[0]), interpolation = cv2.INTER_CUBIC)
        
        i=0
        total = len(self.annotations) -1
        if (int(total/self.partition_size)*self.partition_size == total):
            dumps = int(total/self.partition_size)
        else:
            dumps = int(total/self.partition_size) +1
        remaining = len(self.annotations)
        for idx in tqdm(range(1, len(self.annotations)),desc='Loading Data',leave=True):
            remaining-=1
            line = self.annotations[idx]
            fp, ts, steering, throttle, brake = line.split(",")
            next = load_image(os.path.join(self.root_folder,"raw_images",fp)).astype(np.float32) / 255.0
            next_grayscale = cv2.cvtColor(next,cv2.COLOR_BGR2GRAY)
            next_resize = cv2.resize(next_grayscale, (self.im_size[1], self.im_size[0]), interpolation = cv2.INTER_CUBIC)
            flow = cv2.calcOpticalFlowFarneback(prvs_resize,next_resize, None, 0.5, 3, 20, 8, 5, 1.2, 0)
            self.images[(idx%self.partition_size)-1] = flow.transpose(2, 0, 1)
            self.throttle[(idx%self.partition_size)-1] = float(throttle)
            self.brake[(idx%self.partition_size)-1]=float(brake) 
            self.labels[(idx%self.partition_size)-1] = float(steering)
            prvs_resize = next_resize
            if((idx) % self.partition_size ==0):
                i+=1
                filename = os.path.join(pickle_dir,"saved_image_opticalflow_" + str(i) + ".pkl")
                fp = open(filename, 'wb')
                #tqdm.set_description('Writing Pickle %s'%(filename))
                pickle.dump(self.images, fp, protocol=4)
                fp.close()

                filename = os.path.join(pickle_dir,"saved_labels_opticalflow_" + str(i) + ".pkl")
                fp = open(filename, 'wb')
                #tqdm.set_description('Writing Pickle %s'%(filename))
                pickle.dump(self.labels, fp, protocol=4)
                fp.close()
                
                #tqdm.set_description('Loading Data')

                if(i==(dumps-1)):
                    self.partition_size = remaining 
                self.length = self.partition_size - 1
                self.images = np.tile(0, (self.length,2,self.im_size[0],self.im_size[1])).astype(np.float32)
                self.labels = np.tile(0, (self.length)).astype(np.float64)
                self.throttle = np.tile(0, (self.length)).astype(np.float64)
                self.brake = np.tile(0, (self.length)).astype(np.float64)
            
        if(i<dumps):
            i+=1
            filename =  os.path.join(pickle_dir,"saved_image_opticalflow_" + str(i) + ".pkl")
            fp = open(filename, 'wb')
            #tqdm.set_description('Writing Pickle %s'%(filename))
            pickle.dump(self.images, fp, protocol=4)
            fp.close()

            filename =  os.path.join(pickle_dir,"saved_labels_opticalflow_" + str(i) + ".pkl")
            fp = open(filename, 'wb')
            #tqdm.set_description('Writing Pickle %s'%(filename))
            pickle.dump(self.labels, fp, protocol=4)
            fp.close()
            
            #tqdm.set_description('Loading Data')

        print('%d pickle files saved'%(i))
        self.preloaded=True
    def read_files(self):
        pickle_dir,_ = self.annotation_filepath.split('.')
        pickle_dir+='_data'
        if(not os.path.exists(pickle_dir)):
            os.makedirs(pickle_dir)
        print("loading data")
        i=0
        total = len(self.annotations)
        if (int(total/self.partition_size)*self.partition_size == total):
            dumps = int(total/self.partition_size)
        else:
            dumps = int(total/self.partition_size) +1
        remaining = total
        for (idx,line) in tqdm(enumerate(self.annotations),desc='Loading Data',leave=True):
            remaining-=1
            fp, ts, steering, throttle, brake = line.split(",")
            im = load_image(os.path.join(self.root_folder,"raw_images",fp))
            im = cv2.resize(im, (self.im_size[1], self.im_size[0]), interpolation = cv2.INTER_CUBIC)
            im = np.transpose(im, (2, 0, 1))
            self.images[(idx%self.partition_size)] = im
            self.throttle[(idx%self.partition_size)] = float(throttle)
            self.brake[(idx%self.partition_size)]=float(brake)
            self.labels[(idx%self.partition_size)] = float(steering)
            if((idx) % self.partition_size ==0):
                i+=1
                filename =  os.path.join(pickle_dir,"saved_image_" + str(i) + ".pkl")
                #tqdm.set_description('Writing Pickle %s'%(filename))
                fp = open(filename, 'wb')
                pickle.dump(self.images, fp, protocol=4)
                fp.close()

                filename =  os.path.join(pickle_dir,"saved_labels_" + str(i) + ".pkl")
                #tqdm.set_description('Writing Pickle %s'%(filename))
                fp = open(filename, 'wb')
                pickle.dump(self.labels, fp, protocol=4)
                fp.close()
                
                #tqdm.set_description('Loading Data')

                if(i==(dumps-1)):
                    self.partition_size = remaining
                self.length = self.partition_size
                self.images = np.tile(0, (self.length,3,self.im_size[0],self.im_size[1])).astype(np.int8)
                self.labels = np.tile(0, (self.length)).astype(np.float64)
                self.throttle = np.tile(0, (self.length)).astype(np.float64)
                self.brake = np.tile(0, (self.length)).astype(np.float64)
            
        if(i<dumps):
            i+=1
            filename =  os.path.join(pickle_dir,"saved_image_" + str(i) + ".pkl")
            #tqdm.set_description('Writing Pickle %s'%(filename))
            fp = open(filename, 'wb')
            pickle.dump(self.images, fp, protocol=4)
            fp.close()

            filename =  os.path.join(pickle_dir,"saved_labels_" + str(i) + ".pkl")
            #tqdm.set_description('Writing Pickle %s'%(filename))
            fp = open(filename, 'wb')
            pickle.dump(self.labels, fp, protocol=4)
            fp.close()
        
            #tqdm.set_description('Loading Data')

        print('%d pickle files saved'%(i))
        self.preloaded=True
    def __getitem__(self, index):
        if(self.preloaded):
            im = self.images[index]
            label = self.labels[index]
            throttle = self.throttle[index]
            brake=self.brake[index]
        else:
            if(not self.optical_flow):
                fp, ts, steering, throttle, brake = self.annotations[index].split(",")
                im = load_image(os.path.join(self.root_folder,"raw_images",fp))
                im = cv2.resize(im, (self.im_size[1], self.im_size[0]), interpolation = cv2.INTER_CUBIC)
                im = np.transpose(im, (2, 0, 1))
                label = np.array((float(steering)))
                throttle = np.array(float(throttle))
                brake=np.array(float(brake))
            elif(index!=0):
                pline = self.annotations[index-1]
                pfp, pts, psteering, pthrottle, pbrake = pline.split(",")
                prvs = load_image(os.path.join(self.root_folder,"raw_images",pfp)).astype(np.float32) / 255.0
                prvs_grayscale = cv2.cvtColor(prvs,cv2.COLOR_BGR2GRAY)
                prvs_resize = cv2.resize(prvs_grayscale, (self.im_size[1], self.im_size[0]), interpolation = cv2.INTER_CUBIC)
                fp, ts, steering, throttle, brake = self.annotations[index].split(",")
                next = load_image(os.path.join(self.root_folder,"raw_images",fp)).astype(np.float32) / 255.0
                next_grayscale = cv2.cvtColor(next,cv2.COLOR_BGR2GRAY)
                next_resize = cv2.resize(next_grayscale, (self.im_size[1], self.im_size[0]), interpolation = cv2.INTER_CUBIC)
                flow = cv2.calcOpticalFlowFarneback(prvs_resize,next_resize, None, 0.5, 3, 20, 8, 5, 1.2, 0)
                im= flow.transpose(2, 0, 1)
                label = np.array((float(steering)))
                throttle = np.array(float(throttle))
                brake=np.array(float(brake))
            else:
                blank_image = np.zeros((50,50,3), np.float32)
                prvs_grayscale = cv2.cvtColor(blank_image,cv2.COLOR_BGR2GRAY)
                prvs_resize = cv2.resize(prvs_grayscale, (self.im_size[1], self.im_size[0]), interpolation = cv2.INTER_CUBIC)
                fp, ts, steering, throttle, brake = self.annotations[index].split(",")
                next = load_image(os.path.join(self.root_folder,"raw_images",fp)).astype(np.float32) / 255.0
                next_grayscale = cv2.cvtColor(next,cv2.COLOR_BGR2GRAY)
                next_resize = cv2.resize(next_grayscale, (self.im_size[1], self.im_size[0]), interpolation = cv2.INTER_CUBIC)
                flow = cv2.calcOpticalFlowFarneback(prvs_resize,next_resize, None, 0.5, 3, 20, 8, 5, 1.2, 0)
                im= flow.transpose(2, 0, 1)
                label = np.array((float(steering)))
                throttle = np.array(float(throttle))
                brake=np.array(float(brake))
        if(self.use_float32):
            im = im.astype(np.float32)
            label = label.astype(np.float32)
            throttle = throttle.astype(np.float32)
            brake= brake.astype(np.float32)
        else:
            im = im.astype(np.float64)
            label = label.astype(np.float64)
            throttle = throttle.astype(np.float64)
            brake= brake.astype(np.float64)
        label_tensor = torch.from_numpy(np.array(label))
        img_tensor = torch.from_numpy(im)
        brake_tensor = torch.from_numpy(np.array(brake))
        throttle_tensor = torch.from_numpy(np.array(throttle))
        if(not (self.img_transformation == None)):
            img_tensor = self.img_transformation(img_tensor)
        if(not (self.label_transformation == None)):
            label_tensor = self.label_transformation(label_tensor)
        return img_tensor, throttle_tensor, brake_tensor, label_tensor.view(1)
    def __len__(self):
        return self.length
class F1SequenceDataset(F1Dataset):
    def __init__(self, root_folder, annotation_filepath, im_size,\
        context_length = 25, sequence_length=25, use_float32=False, img_transformation = None, label_transformation = None, optical_flow = False):
        super(F1SequenceDataset, self).__init__(root_folder, annotation_filepath, im_size, use_float32=use_float32, img_transformation = img_transformation, label_transformation = label_transformation, optical_flow=optical_flow)
        self.sequence_length = sequence_length
        self.context_length = context_length
        self.length -= (context_length + sequence_length)
    def load_image_seq(self,index,label_start):
        self.images=[]
        
        return self.images
    def __getitem__(self, index):
        flag=True
        label_start = index + self.context_length
        label_end = label_start + self.sequence_length
        if(self.preloaded):     
            previous_control = self.labels[index:label_start]
            seq = self.images[index:label_start]
            seq_throttle = self.throttle[index:label_start]
            seq_brake = self.brake[index:label_start]
            seq_labels = self.labels[label_start:label_end]
            if(len(seq_labels)!= self.sequence_length):
                flag=False
                actual = len(seq_labels)
                label_start = 0 + self.context_length
                label_end = label_start + self.sequence_length
                previous_control = self.labels[0:label_start]
                seq = self.images[0:label_start]
                seq_throttle = self.throttle[0:label_start]
                seq_brake = self.brake[0:label_start]
                seq_labels = self.labels[label_start:label_end]
                #print('adjusted sequence used for %d batch (%d,%d)'%(index,actual,self.sequence_length))
        else:
            seq=[]   
            previous_control=[]
            prvs_resize =None
            for idx in range(index,label_start):
                if(not self.optical_flow):
                    fp, ts, steering, throttle, brake = self.annotations[idx].split(",")
                    im = load_image(os.path.join(self.root_folder,"raw_images",fp))
                    im = cv2.resize(im, (self.im_size[1], self.im_size[0]), interpolation = cv2.INTER_CUBIC)
                    im = np.transpose(im, (2, 0, 1))
                    seq.append(im)
                else:
                    if(prvs_resize is None):
                        pline = self.annotations[idx]
                        pfp, pts, psteering, pthrottle, pbrake = pline.split(",")
                        prvs = load_image(os.path.join(self.root_folder,"raw_images",pfp)).astype(np.float32) / 255.0
                        prvs_grayscale = cv2.cvtColor(prvs,cv2.COLOR_BGR2GRAY)
                        prvs_resize = cv2.resize(prvs_grayscale, (self.im_size[1], self.im_size[0]), interpolation = cv2.INTER_CUBIC)
                    fp, ts, steering, throttle, brake = self.annotations[idx+1].split(",")
                    next = load_image(os.path.join(self.root_folder,"raw_images",fp)).astype(np.float32) / 255.0
                    next_grayscale = cv2.cvtColor(next,cv2.COLOR_BGR2GRAY)
                    next_resize = cv2.resize(next_grayscale, (self.im_size[1], self.im_size[0]), interpolation = cv2.INTER_CUBIC)
                    flow = cv2.calcOpticalFlowFarneback(prvs_resize,next_resize, None, 0.5, 3, 20, 8, 5, 1.2, 0)
                    im= flow.transpose(2, 0, 1)
                    seq.append(im)
                    self.brake[idx] = brake
                    self.throttle[idx] = throttle
                    previous_control.append(steering)
                    prvs_resize=next_resize
            for idx in range(label_start,label_end):
                fp, ts, steering, throttle, brake = self.annotations[idx].split(",")
                self.labels[idx] = steering
            seq = np.asarray(seq,dtype=np.float32)
            seq_throttle = self.throttle[index:label_start]
            seq_brake = self.brake[index:label_start]
            seq_labels = self.labels[label_start:label_end]
            previous_control = np.asarray(previous_control,dtype=np.float32)   
            #raise NotImplementedError("Must preload images for sequence dataset")
        if(self.use_float32):
            seq = seq.astype(np.float32)
            seq_throttle = seq_throttle.astype(np.float32)
            seq_brake = seq_brake.astype(np.float32)
            seq_labels = seq_labels.astype(np.float32)
            previous_control = previous_control.astype(np.float32)
        else:
            seq = seq.astype(np.float64)
            seq_throttle = seq_throttle.astype(np.float64)
            seq_brake = seq_brake.astype(np.float64)
            seq_labels = seq_labels.astype(np.float64)
            previous_control = previous_control.astype(np.float64)
        label_tensor = torch.from_numpy(seq_labels)
        previous_control_tensor = torch.from_numpy(previous_control)
        img_tensor = torch.from_numpy(seq)
        throttle_tensor = torch.from_numpy(seq_throttle)
        brake_tensor = torch.from_numpy(seq_brake)
        if(not (self.img_transformation == None)):
            for i in range(0, img_tensor.shape[0]):
                img_tensor[i]=self.img_transformation(img_tensor[i])
        if(not (self.label_transformation == None)):
            for i in range(0, label_tensor.shape[0]):
                label_tensor[i]=self.label_transformation(label_tensor[i])
                previous_control_tensor[i]=self.label_transformation(previous_control_tensor[i])
        
        
        return img_tensor,throttle_tensor, brake_tensor, previous_control_tensor.view(self.context_length,1), label_tensor.view(self.sequence_length,1),flag
    def __len__(self):
        return self.length