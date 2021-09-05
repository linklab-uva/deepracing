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
import time

def packetSortKey(packet):
    return packet.timestamp
def extractROI(x, y, w, h, image):
    return image[y:y+h, x:x+w].copy()
def main(args):
    img_folder = args.image_dir
    img_ext = args.imext
    cropped_dir = os.path.join(img_folder, "cropped_images")
    if os.path.isdir(cropped_dir):
        shutil.rmtree(cropped_dir, ignore_errors=True)
    print("Getting image files from disk", flush=True)
    img_files = []
    keys = []
    for f in os.listdir(img_folder):
        if not os.path.isfile(os.path.join(img_folder,f)):
            continue
        key, ext = os.path.splitext(f)
        if ext.lower()==img_ext:
            keys.append(key)
            img_files.append(os.path.join(img_folder,f))

    im_size = None
    imrows = args.imrows
    imcols = args.imcols
    roi = args.ROI
    dbpath = os.path.join(img_folder,"lmdb")
    if(os.path.isdir(dbpath)):
        if args.override:
            shutil.rmtree(dbpath)
            time.sleep(1.0)
            os.makedirs(dbpath)
        else:
            print("Database folder " + dbpath+ " already exists. add --override if you wish to delete this and rebuild the dataset.", flush=True)
            exit(0)
    im = deepracing.imutils.readImage(img_files[0])
    db = deepracing.backend.ImageLMDBWrapper()
    windowname = "Test Image"
    if roi is not None:
        assert(len(roi) == 4)
        r = int(roi[0])
        c = int(roi[1])
        fr = float(roi[2])
        fc = float(roi[3])
        topleft = np.asarray([r,c], dtype=np.int64)
        cropfactors = np.asarray([fr, fc], dtype=np.float64)
        cropsize = np.round(cropfactors*(np.asarray(im.shape[0:2], dtype=np.float64)-topleft.astype(np.float64))).astype(np.int64)
    else:    
        factor = args.display_resize_factor
        cv2.namedWindow(windowname,cv2.WINDOW_AUTOSIZE)
        c_,r_,w_,h_ = cv2.selectROI(windowname, cv2.cvtColor(deepracing.imutils.resizeImageFactor(im,factor), cv2.COLOR_RGB2BGR), showCrosshair =True)
        #print((c_,r_,w_,h_))
        r = min(int(round(r_/factor)),im.shape[0])
        c = min(int(round(c_/factor)),im.shape[1])
        h = min(int(round(h_/factor)),im.shape[0])
        w = min(int(round(w_/factor)),im.shape[1])
        cropsize = np.asarray([h,w], dtype=np.int64)
        topleft = np.asarray([r,c], dtype=np.int64)
        cropfactors = cropsize.astype(np.float64)/(np.asarray(im.shape[0:2], dtype=np.float64)-topleft.astype(np.float64))
        print("Selected ROI:", flush=True)
        print((r,c,h,w), flush=True)
    try:
        cv2.imshow(windowname, cv2.cvtColor(im[topleft[0]:topleft[0]+cropsize[0], topleft[1]:topleft[1]+cropsize[1]], cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyWindow(windowname)
    except:
        pass
    print("Input Image Size: " + str(im.shape[0:2]), flush=True)
    print("Top-Left: " + str(topleft), flush=True)
    print("Crop Factors: " + str(cropfactors), flush=True)
    print("Crop Size: " + str(cropsize), flush=True)

    if(imrows>0 and imcols>0):
        im_size = np.array((imrows, imcols, im.shape[2]))
    else:
        im_size = np.array( ( h, w, im.shape[2] ) )
    if(args.mapsize>0):
        mapsize = int(args.mapsize)
    else:
        mapsize = int( float(np.prod(im_size) + 12 )*float(len(img_files))*1.1 )
    print("Using a mapsize of " + str(mapsize), flush=True)
    if not (os.path.isdir(dbpath)):
        os.makedirs(dbpath)
    with open(os.path.join(dbpath,"config.yaml"),"w") as f:
        yaml.dump({"original_image_size": list(im.shape[0:2]),"image_size": im_size[0:2].tolist(), "topleft": topleft.tolist(), "cropfactors": cropfactors.tolist()}, f, Dumper=yaml.SafeDumper)
    db.readImages(img_files, keys, dbpath, im_size[0:2].tolist(), ROI=(topleft[0],topleft[1],cropsize[0],cropsize[1]), mapsize=mapsize)
    print("Done creating LMDB", flush=True)
    db.readDatabase(dbpath, mapsize=mapsize, max_spare_txns=32)
    windowname="DB Image"
    idx = random.randint(0,len(keys)-1)
    #randomkey = keys[idx]
    randomkey = random.choice(keys)
    print("Grabbing image with key: %s" %(randomkey,), flush=True)
    imtuple = db.getImage(randomkey)
    imtimestamp = imtuple[0]
    im = imtuple[1]
    print("Image has size: %s and timestamp %f" %(str(im.shape),imtimestamp), flush=True)
    try:
        plt.imshow(im)
        plt.show()
        # cv2.imshow(windowname, cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
        # cv2.waitKey(0)
        # cv2.destroyWindow(windowname)
    except Exception as ex:
        print(im, flush=True)
        print(im.shape, flush=True)
        print("Could not display db image because:", flush=True)
        print(ex, flush=True)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Load an image directory into a database")
    parser.add_argument("image_dir", type=str, help="Directory containing the images")
    parser.add_argument("imrows", type=int, help="Number of rows to resize images to")
    parser.add_argument("imcols", type=int, help="Number of cols to resize images to")
    parser.add_argument("--display_resize_factor", type=float, default=0.5, help="Resize the first image by this factor for selecting a ROI.")
    parser.add_argument("--mapsize", type=int, default=-1, help="Map size for the LMDB.")
    parser.add_argument('-R','--ROI', nargs=4, help='Region of Interest (ROI) to capture of the form [r, c, fr, fc] with (r,c) being the top-left corner of the ROI. fr fc are the ratios (in row and column dimension) of the remainder of the image to grab', default=None)
    parser.add_argument('--override', action="store_true", help='Delete existing DB if it already exists', default=None)
    parser.add_argument('--imext', type=str, default=".jpg", help='Load image files with this extension')
    args = parser.parse_args()
    main(args)
