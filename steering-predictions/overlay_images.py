from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import cv2
import img_utils.utils as imutils
import numpy as np
import argparse
import os
from tqdm import tqdm as tqdm
def main():
    parser = argparse.ArgumentParser(description="Deepf1 playground")
    parser.add_argument("--x", type=int, default=25,  help="X coordinate for where to put the center of the steering wheel")
    parser.add_argument("--y", type=int, default=25,  help="Y coordinate for where to put the center of the steering wheel")
    parser.add_argument("--wheelrows", type=int, default=50,  help="Number of rows to resize the steering wheel to")
    parser.add_argument("--wheelcols", type=int, default=50,  help="Number of columns to resize the steering wheel to")
    parser.add_argument("--max_angle", type=float, default=180.0,\
          help="Maximum angle that the scaled annotations represent")
    parser.add_argument("--output_video", type=str, default='annotated_video.avi',\
          help="Output video file")
    args = parser.parse_args()
    output_video = args.output_video
    wheelrows = args.wheelrows
    wheelcols = args.wheelcols
    x = int(args.x-wheelrows/2)
    y = int(args.y-wheelcols/2)
    max_angle = args.max_angle
    annotations_path = os.path.join('slow_run_australia_track2','cubic_interpolation.csv')
    annotations_file = open(annotations_path,'r')
    wheel = cv2.imread('steering_wheel.png', cv2.IMREAD_UNCHANGED)  
    wheel =  cv2.resize(wheel, (wheelrows, wheelcols)) 
    input_folder = os.path.join('slow_run_australia_track2','raw_images')
    output_folder = os.path.join('slow_run_australia_track2','overlayed_images')
    annotations = annotations_file.readlines()
    filename, _, anglestr = annotations[0].split(",")
    img_path = os.path.join(input_folder,filename)
    background = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    size = background.shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_vid = os.path.join('slow_run_australia_track2','overlayed_images',output_video)
    videoout = cv2.VideoWriter(output_vid ,fourcc, 60.0, (size[1], size[0]),True)
    for annotation in tqdm(annotations):
        filename, ts, anglestr = annotation.split(",")
        img_path = os.path.join(input_folder,filename)
        background = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        anglestr = anglestr.replace("\n","")
        angle = float(anglestr)
        scaled_angle = max_angle * angle
        M = cv2.getRotationMatrix2D((wheelrows/2,wheelcols/2),scaled_angle,1)
        wheel_rotated = cv2.warpAffine(wheel,M,(wheelrows,wheelcols))
        overlayed = imutils.overlay_image(background,wheel_rotated,x,y)
        name, _ = filename.split(".")
        _,_,img_num_str = name.split("_")
        img_num_str = img_num_str.replace("\n","")
        output_path = os.path.join(output_folder,'overlayed_image_' + img_num_str + ".jpg")
        cv2.imwrite(output_path,overlayed)
        videoout.write(overlayed)

if __name__ == '__main__':
    main()

