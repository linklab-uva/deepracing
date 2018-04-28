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
    parser.add_argument("--x", type=int, default=0,  help="X coordinate of uppper-left corner of box to capture")
    parser.add_argument("--y", type=int, default=0,  help="Y coordinate of uppper-left corner of box to capture")
    args = parser.parse_args()
    x = args.x
    y = args.y
    img_path = os.path.join('slow_run_australia_track2','raw_images','raw_image_57.jpg')
    annotations_path = os.path.join('slow_run_australia_track2','zeroth_degree_interpolation.csv')
    annotations_file = open(annotations_path,'r')
    annotations = annotations_file.readlines()
    print(annotations[50].split(","))
    print(len(annotations))
    print(img_path)
    background = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    wheel = cv2.imread('steering_wheel.png', cv2.IMREAD_UNCHANGED)  
    cv2.namedWindow('Display image')          ## create window for display
    imutils.overlay_image(background,wheel,x,y)
    cv2.imshow('Display image', background)          ## Show image in the window
    cv2.waitKey(0)

    

if __name__ == '__main__':
    main()

