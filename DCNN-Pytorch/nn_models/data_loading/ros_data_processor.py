import numpy as np
from PIL import Image
import argparse
import os
from tqdm import tqdm as tqdm
import csv
from operator import itemgetter
import sys
import cv2
import codecs

def main():

    csv.field_size_limit(2000000000)
    parser = argparse.ArgumentParser(description="Data Augmentor")
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--clip", type=int, default=0,required=False)
    args = parser.parse_args()
    directory = args.path
    clip = args.clip
    annot_file = os.path.join(directory,"_slash_training_data.csv")
    output_path = os.path.join(directory,"raw_images")
    data_file=os.path.join(directory,"data.csv")
    final_data=[]
    seen={}
    count=0
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
                if(ts not in seen):
                    seen[ts]=steer
                else:
                    continue
            
                image = image[1:-1]
                i_data = image.split(',')
                i_data = list(map(int, i_data))
                img_data = np.asarray(i_data,np.uint8).reshape(480,640,3)
                final_image = Image.fromarray(img_data.astype('uint8'), 'RGB')
            
                out_dir = os.path.join(output_path,str(ts)+'.jpeg')
                final_image.save(out_dir)
                count+=1
                final_data.append([str(ts)+'.jpeg',str(ts),str(steer),str(throttle),str(0)])
    
        with open(data_file,'w',newline='') as f:
            writer = csv.writer(f)
            if(clip):
                writer.writerows(final_data[:len(final_data)-clip])
            else:
                writer.writerows(final_data)

    imgfiles = set([f for f in os.listdir(output_path) if os.path.isfile(os.path.join(output_path, f))])
    with open(data_file,'r',newline='') as f:
            reader = csv.reader(f)           
            for row in reader:
                img_path = row[0]
                if(os.path.exists(os.path.join(output_path,img_path)) and img_path in imgfiles):
                    imgfiles.remove(img_path)
    for img in tqdm(imgfiles,desc='Clipping Data:'):
        os.remove(os.path.join(output_path,img))                
if __name__ == '__main__':
    main()
