import cv2
import numpy as np
import readline
import os
import argparse
def sortkey(s: str):
    return int(s.split("_")[-1].split(".")[0])
def make_video(images, outimg=None, fps=2, size=None, is_color=True, format="MJPG", outvid='image_video.avi'):
    #from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
    fourcc = cv2.VideoWriter_fourcc(*format)
    
    vid = None
    for image in images:
        if not os.path.exists(image):
            raise FileNotFoundError(image)
        img = cv2.imread(image)
        basename = os.path.basename(image)
        if vid is None:
            if size is None:
                size = img.shape[1], img.shape[0]
            vid = cv2.VideoWriter(outvid, fourcc, float(fps), size, is_color)
        cv2.putText(img,basename, tuple(np.round(np.array(((size[0]/2 - 15*len(basename),size[1]/2)))).astype(np.int32)), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0))
        vid.write(img)
    vid.release()
    return vid
parser = argparse.ArgumentParser(description='Turn a bunch of images into a video.')
parser.add_argument('--folder', type=str, required=True)
args = parser.parse_args()
folder = args.folder
files = [os.path.join(folder,f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) and str.lower(os.path.join(folder, f).split(".")[-1])=="jpg"]
print(files)
filessorted = sorted(files,key=sortkey)
print(filessorted)
video = make_video(filessorted,fps=60)