import numpy as np
import data_loading.backend
import os
import argparse
import skimage
import skimage.io
import shutil
def main():
    parser = argparse.ArgumentParser(description="Load an image directory into a database")
    parser.add_argument("image_dir", type=str, help="Directory containing the images")
    parser.add_argument("imrows", type=int, help="Number of rows to resize images to")
    parser.add_argument("imcols", type=int, help="Number of cols to resize images to")
    args = parser.parse_args()
    img_folder = args.image_dir
    keys = [fname for fname in  os.listdir(img_folder) if os.path.isfile(os.path.join(img_folder,fname)) and os.path.splitext(fname)[1]==".jpg"]
    img_files = [os.path.join(img_folder, key) for key in keys]
    im_size = None
    imrows = args.imrows
    imcols = args.imcols
    im = skimage.io.imread(img_files[0])
    if imrows>0 and imcols>0:
        im_size = np.array((args.imrows, args.imcols, im.shape[2]))
    dbpath = os.path.join(img_folder,"lmdb")
    if(os.path.isdir(dbpath)):
        s=""
        while not (s=='n' or s=='y'):
            input("Database folder " + dbpath+ " already exists. overwrite with new data? [y/n]\n")
        if(s=='n'):
            print("Goodbye then!")
            exit(0)
        shutil.rmtree(dbpath)
    db = data_loading.backend.LMDBWrapper()
    db.readImages(img_files, keys, dbpath, im_size=im_size)
if __name__ == '__main__':
  main()
