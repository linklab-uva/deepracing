import cv2
import csv
import os
import argparse

def main():

    parser = argparse.ArgumentParser(description="Grab frames from videos and generate fake annotation")
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--name", type=str, required=True)
    args = parser.parse_args()

    dir = args.path
    name=args.name

    raw_dir = os.path.join(dir,"raw_images")
    raw_full_dir = os.path.join(dir,"raw_images_full")
    vid_file = os.path.join(dir,name+'.mp4')
    annot_file = os.path.join(dir,name+'.csv')

    if(not os.path.isdir(raw_dir)):
        os.mkdir(raw_dir)
    if(not os.path.isdir(raw_full_dir)):
        os.mkdir(raw_full_dir)
    video = cv2.VideoCapture(vid_file)
    count=1
     
    new_data=[]
    while count<3438: 
        ret, frame = video.read()
        cv2.imwrite(os.path.join(raw_full_dir,str(count)+'.jpeg'),frame)
        cv2.imwrite(os.path.join(raw_dir,str(count)+'.jpeg'),frame[0:250,:])
        new_data.append([str(count)+'.jpeg',count,str(0),str(0),str(0)])
        count+=1
    with open(annot_file,'w',newline='') as f:
        writer = csv.writer(f)
        writer.writerows(new_data)

if __name__ == '__main__':
    main()
