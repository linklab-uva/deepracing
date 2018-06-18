import numpy as np
from PIL import Image
import argparse
import os
from tqdm import tqdm as tqdm
import csv
from operator import itemgetter
import sys
import cv2

def main():

    csv.field_size_limit(1000000000)
    parser = argparse.ArgumentParser(description="Data Augmentor")
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--clip", type=int, default=0,required=False)
    args = parser.parse_args()
    directory = args.path
    clip = args.clip
    annot_file = os.path.join(directory,"_slash_training_data.csv")
    output_path = os.path.join(directory,"raw_images")
    final_data=[]
    if(os.path.exists(output_path)==False):
        os.mkdir(output_path)
        with open(annot_file,'r',newline='') as f:
            reader = csv.reader(f)
            next(reader)
            for row in tqdm(reader,desc='Processing training Data:'):
                image=row[13]
                ts=row[3]
                steer = 2*float(row[22])-1
                throttle = float(row[16])
            
                image = image[1:-1]
                i_data = image.split(',')
                i_data = list(map(int, i_data))
                img_data = np.asarray(i_data,np.uint8).reshape(480,640,3)
                final_image = Image.fromarray(img_data.astype('uint8'), 'RGB')
            
                out_dir = os.path.join(output_path,str(ts)+'.jpeg')
                final_image.save(out_dir)
                final_data.append([out_dir,str(ts),str(steer),str(throttle),str(0)])
    
    data_file=os.path.join(directory,"data.csv")
    with open(data_file,'w',newline='') as f:
        writer = csv.writer(f)
        if(clip):
            writer.writerows(final_data[:len(final_data)-clip])
        else:
            writer.writerows(final_data)

    imgfiles = set([f for f in os.listdir(mypath) if isfile(join(mypath, f))])
    with open(data_file,'r',newline='') as f:
            reader = csv.reader(f)           
            for row in reader:
                img_path = row[0]
                imgfiles.remove(img_path)
    for img in tqdm(imgfiles,builtin_method_descriptor='Clipping Data:'):
        os.remove(img)                    
if __name__ == '__main__':
    main()
