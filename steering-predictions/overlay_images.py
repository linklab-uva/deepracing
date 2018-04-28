from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import cv2
import img_utils.utils as imutils
import numpy as np
import argparse
import os
def main():
    parser = argparse.ArgumentParser(description="Deepf1 playground")
    parser.add_argument("--x", type=int, default=0,  help="X coordinate for where to put the steering wheel")
    parser.add_argument("--y", type=int, default=0,  help="Y coordinate for where to put the steering wheel")
    args = parser.parse_args()
    x = args.x
    y = args.y
    img_path = os.path.join('slow_run_australia_track2','raw_images','raw_image_57.jpg')
    annotations_path = os.path.join('slow_run_australia_track2','zeroth_degree_interpolation.csv')
    annotations_file = open(annotations_path,'r')
    annotations = annotations_file.readlines()
    background = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    wheel = cv2.imread('steering_wheel.png', cv2.IMREAD_UNCHANGED)  
    output_folder = os.path.join('slow_run_australia_track2','overlayed_images')
    filename, ts, anglestr = annotations[0].split(",")
    anglestr = anglestr.replace("\n","")
    cv2.namedWindow('Display image')          ## create window for display
    imutils.overlay_image(background,wheel,x,y)
    cv2.imshow('Display image', background)          ## Show image in the window
    cv2.waitKey(0)
    angle = float(anglestr)
    print(angle)
    name, ext = filename.split(".")
    _,_,img_num_str = name.split("_")
    img_num_str = img_num_str.replace("\n","")
    print(img_num_str)
    output_path = os.path.join(output_folder,'overlayed_image_' + img_num_str + ".jpg")
    cv2.imwrite(output_path,background)
    

if __name__ == '__main__':
    main()

