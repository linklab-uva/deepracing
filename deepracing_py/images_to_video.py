import cv2
import numpy as np
import readline
import os
import argparse
from tqdm import tqdm as tqdm
def sortkey(s: str):
    return int(s.split("_")[-1].split(".")[0])
def make_video(images, fps, outimg=None, size=None, is_color=True, format="MJPG", outvid='image_video.avi'):
    #from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
    fourcc = cv2.VideoWriter_fourcc(*format)
    
    vid = None
    for image in tqdm(images):
        if not os.path.exists(image):
            raise FileNotFoundError(image)
        img = cv2.imread(image, flags=cv2.IMREAD_COLOR)
        basename = os.path.basename(image)
        if vid is None:
            if size is None:
                size_ = img.shape[1], img.shape[0]
            else:
                size_ = size
            vid = cv2.VideoWriter(outvid, fourcc, fps, size_, is_color)
        #cv2.putText(img,basename, tuple(np.round(np.array(((size[0]/2 - 15*len(basename),size[1]/2)))).astype(np.int32)), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0))
        vid.write(img)
    vid.release()
    return vid
parser = argparse.ArgumentParser(description='Turn a bunch of images into a video.')
parser.add_argument('--folder', type=str, required=True)
parser.add_argument('--output_file', type=str, default="image_video.avi", required=False)
parser.add_argument('--fps', type=float, default=60, required=False)
args = parser.parse_args()
folder = args.folder
files = [os.path.join(folder,f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) and str.lower(os.path.join(folder, f).split(".")[-1])=="png"]
print("Got image files, sorting by file index")
filessorted = sorted(files,key=sortkey)
print("Making video.")
video = make_video(filessorted, args.fps, outvid=args.output_file)