import numpy as np
import os
import argparse
import skimage
import skimage.io
import shutil
import deepracing.imutils
import deepracing.backend
import cv2
import deepracing.imutils
import random
import yaml
from deepracing.protobuf_utils import getAllImageFilePackets
from tqdm import tqdm as tqdm
from matplotlib import pyplot as plt
def packetSortKey(packet):
    return packet.timestamp
def extractROI(x, y, w, h, image):
    return image[y:y+h, x:x+w].copy()
def main(args):
    img_folder = args.image_dir
    print("Getting image files from disk")
    packets = sorted(getAllImageFilePackets(img_folder, args.json), key=packetSortKey)
    img_files = [os.path.join(img_folder,packet.image_file) for packet in packets]
    keys = [os.path.splitext(os.path.basename(img_file))[0] for img_file in img_files]

    im_size = None
    imrows = args.imrows
    imcols = args.imcols
    roi = args.ROI
    dbpath = os.path.join(img_folder,"image_lmdb")
    if(os.path.isdir(dbpath)):
        if args.override:
            shutil.rmtree(dbpath)
            overwrite_images = True
        else:
            s=""
            while not (s=='n' or s=='y'):
                s=input("Database folder " + dbpath+ " already exists. overwrite with new data? [y/n]\n")
            if(s=='y'):
                shutil.rmtree(dbpath)
                overwrite_images = True
            else:
                print("Skipping image db")
                overwrite_images = False
    else:
        overwrite_images = True
    im = deepracing.imutils.readImage(img_files[0])
    db = deepracing.backend.ImageLMDBWrapper()
    
    if overwrite_images:
        if roi is not None:
            assert(len(roi) == 4)
            x = int(roi[0])
            y = int(roi[1])
            w = int(roi[2])
            h = int(roi[3])
        else:    
            factor = args.display_resize_factor
            windowname = "Test Image"
            cv2.namedWindow(windowname,cv2.WINDOW_AUTOSIZE)
            x_,y_,w_,h_ = cv2.selectROI(windowname, cv2.cvtColor(deepracing.imutils.resizeImageFactor(im,factor), cv2.COLOR_RGB2BGR), showCrosshair =True)
            #print((x_,y_,w_,h_))
            x = int(round(x_/factor))
            y = int(round(y_/factor))
            w = int(round(w_/factor))
            h = int(round(h_/factor))
            print("Selected ROI:")
            print((x,y,w,h))
            cv2.imshow(windowname, cv2.cvtColor(im[y:y+h, x:x+w], cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)
            cv2.destroyWindow(windowname)
        if(imrows>0 and imcols>0):
            im_size = np.array((imrows, imcols, im.shape[2]))
        else:
            im_size = np.array( ( h, w, im.shape[2] ) )
        if(args.mapsize>0):
            mapsize = int(args.mapsize)
        else:
            mapsize = int( float(np.prod(im_size) + 12 )*float(len(img_files))*1.1 )
        print("Using a mapsize of " + str(mapsize))
        db.readImages(img_files, keys, dbpath, im_size, ROI=(x,y,w,h), mapsize=mapsize)
        with open(os.path.join(dbpath,"config.yaml"),"w") as f:
            yaml.dump({"ROI":[x,y,w,h]},f,Dumper=yaml.SafeDumper)
        print("Done creating LMDB")
        db.readDatabase(dbpath, mapsize=mapsize, max_spare_txns=32)
    else:
        im_size = np.array((imrows, imcols, im.shape[2]))
        if(args.mapsize>0):
            mapsize = int(args.mapsize)
        else:
            mapsize = int( float(np.prod(im_size) + 12 )*float(len(img_files))*1.1 )
        db.readDatabase(dbpath, mapsize=mapsize, max_spare_txns=32)
    windowname="DB Image"
    idx = random.randint(0,len(keys)-1)
    #randomkey = keys[idx]
    randomkey = "image_626"
    print("Grabbing image with key: %s" %(randomkey))
    im = db.getImage(randomkey)
    try:
        plt.imshow(im)
        plt.show()
        # cv2.imshow(windowname, cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
        # cv2.waitKey(0)
        # cv2.destroyWindow(windowname)
    except Exception as ex:
        print(im)
        print(im.shape)
        print("Could not display db image because:")
        print(ex)
    if args.optical_flow:
        optflow_db = deepracing.backend.OpticalFlowLMDBWrapper()
        optflow_dbpath = os.path.join(img_folder,"optical_flow_lmdb")
        if(os.path.isdir(optflow_dbpath)):
            s=""
            while not (s=='n' or s=='y'):
                s=input("Database folder " + optflow_dbpath+ " already exists. overwrite with new data? [y/n]\n")
            if(s=='n'):
                print("Goodbye then!")
                exit(0)
            shutil.rmtree(optflow_dbpath)
        if(args.mapsize>0):
            mapsize = int(8*args.mapsize/3)
        else:
            mapsize = int( float(np.prod(im_size[0:2])*8 + 12 )*float(len(img_files))*1.1 )
        print("Using an optical flow mapsize of " + str(mapsize))
        optflow_db.readImages( keys, optflow_dbpath, db, mapsize=mapsize )
        optflow_db.readDatabase(optflow_dbpath, mapsize=mapsize)
        try:
            r, c, _ = im.shape
            gd = 8 # every 10th point
            X = np.linspace(0,c-1,int(c/gd)+int(bool(not c%gd==0)))
            Y = np.linspace(0,r-1,int(r/gd)+int(bool(not r%gd==0)))
            flow_db = optflow_db.getImage(randomkey)
            print(flow_db.shape)
            dY = flow_db[::gd,::gd,0]
            dX = flow_db[::gd,::gd,1]
            print(X.shape)
            print(Y.shape)
            print(dX.shape)
            print(dY.shape)
            plt.figure()
            plt.quiver(X, Y, dX, dY, color='r')
            plt.imshow(im)
            plt.show()
        except Exception as e:
            print(e)
            print("Loading of optical flow database was successful, but could not visualize the optical flow image for reason printed above")
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Load an image directory into a database")
    parser.add_argument("image_dir", type=str, help="Directory containing the images")
    parser.add_argument("imrows", type=int, help="Number of rows to resize images to")
    parser.add_argument("imcols", type=int, help="Number of cols to resize images to")
    parser.add_argument("--display_resize_factor", type=float, default=0.5, help="Resize the first image by this factor for selecting a ROI.")
    parser.add_argument("--mapsize", type=float, default=-1.0, help="Map size for the LMDB.")
    parser.add_argument('-R','--ROI', nargs='+', help='ROI to capture', default=None)
    parser.add_argument('--optical_flow', action="store_true", help='Compute optical flow as well', default=None)
    parser.add_argument('--json', action="store_true", help='Use json packets', default=None)
    parser.add_argument('--override', action="store_true", help='Delete existing DB if it already exists', default=None)
    args = parser.parse_args()
    main(args)
