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
from functools import partials
import random
from deepracing.pose_utils import getAllImageFilePackets
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
    im = deepracing.imutils.readImage(img_files[0])
    if roi is not None:
        assert(len(roi) == 4)
        x = int(roi[0])
        y = int(roi[1])
        w = int(roi[2])
        h = int(roi[3])
        f = partial(extractROI,x,y,w,h)
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
        f = partial(extractROI,x,y,w,h)
        cv2.imshow(windowname, cv2.cvtColor(f(im), cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyWindow(windowname)
    if(imrows>0 and imcols>0):
        im_size = np.array((imrows, imcols, im.shape[2]))
    else:
        im_size = np.array( ( h, w, im.shape[2] ) )
    dbpath = os.path.join(img_folder,"image_lmdb")
    if(os.path.isdir(dbpath)):
        if args.override:
            shutil.rmtree(dbpath)
        else:
            s=""
            while not (s=='n' or s=='y'):
                s=input("Database folder " + dbpath+ " already exists. overwrite with new data? [y/n]\n")
            if(s=='n'):
                print("Goodbye then!")
                exit(0)
            shutil.rmtree(dbpath)
    if(args.mapsize>0):
        mapsize = int(args.mapsize)
    else:
        mapsize = int( float(np.prod(im_size) + 12 )*float(len(img_files))*1.1 )
    print("Using a mapsize of " + str(mapsize))
    db = deepracing.backend.ImageLMDBWrapper()
    db.readImages(img_files, keys, dbpath, im_size, func=f, mapsize=mapsize)
    print("Done creating LMDB")
    db.readDatabase(dbpath, mapsize=mapsize, max_spare_txns=16)
    windowname="DB Image"
    idx = random.randint(0,len(keys)-1)
    randomkey = keys[idx]
    print("Grabbing image with key: %s" %(randomkey))
    im = db.getImage(randomkey)
    try:
        cv2.imshow(windowname, cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyWindow(windowname)
    except Exception as ex:
        print(im)
        print(im.shape)
        print("Could not display db image because:")
        print(ex)
    if args.optical_flow:
        optflow_dbpath = os.path.join(img_folder,"optical_flow_lmdb")
        if(os.path.isdir(optflow_dbpath)):
            s=""
            while not (s=='n' or s=='y'):
                s=input("Database folder " + optflow_dbpath+ " already exists. overwrite with new data? [y/n]\n")
            if(s=='n'):
                print("Goodbye then!")
                exit(0)
            shutil.rmtree(optflow_dbpath)
        optflow_db = deepracing.backend.OpticalFlowLMDBWrapper()
        if(args.mapsize>0):
            mapsize = int(8*args.mapsize/3)
        else:
            mapsize = int( float(np.prod(im_size[0:2])*8 + 12 )*float(len(img_files))*1.1 )
        print("Using an optical flow mapsize of " + str(mapsize))
        optflow_db.readImages( keys, optflow_dbpath, db, mapsize=mapsize )
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
