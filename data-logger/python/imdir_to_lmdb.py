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
from functools import partial
def extractROI(x, y, w, h, image):
    return image[y:y+h, x:x+w].copy()
def main():
    parser = argparse.ArgumentParser(description="Load an image directory into a database")
    parser.add_argument("image_dir", type=str, help="Directory containing the images")
    parser.add_argument("imrows", type=int, help="Number of rows to resize images to")
    parser.add_argument("imcols", type=int, help="Number of cols to resize images to")
    parser.add_argument("--display_resize_factor", type=float, default=0.5, help="Resize the first image by this factor for selecting a ROI.")
    parser.add_argument("--mapsize", type=float, default=1e10, help="Map size for the LMDB.")
    parser.add_argument('-R','--ROI', nargs='+', help='ROI to capture', default=None)
    args = parser.parse_args()
    img_folder = args.image_dir
    keys = [os.path.splitext(fname)[0] for fname in  os.listdir(img_folder) if os.path.isfile(os.path.join(img_folder,fname)) and (os.path.splitext(fname)[1]==".jpg" or os.path.splitext(fname)[1]==".png")]
    img_files = [os.path.join(img_folder,fname) for fname in  os.listdir(img_folder) if os.path.isfile(os.path.join(img_folder,fname)) and (os.path.splitext(fname)[1]==".jpg" or os.path.splitext(fname)[1]==".png")]
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
        s=""
        while not (s=='n' or s=='y'):
            s=input("Database folder " + dbpath+ " already exists. overwrite with new data? [y/n]\n")
        if(s=='n'):
            print("Goodbye then!")
            exit(0)
        shutil.rmtree(dbpath)
    db = deepracing.backend.ImageLMDBWrapper()
    db.readImages(img_files, keys, dbpath, im_size, func=f, mapsize=args.mapsize)
if __name__ == '__main__':
  main()
