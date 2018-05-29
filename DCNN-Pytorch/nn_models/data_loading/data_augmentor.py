import numpy as np
import cv2
import argparse
import os
from tqdm import tqdm as tqdm

def load_image(filepath):
    img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    return img

def main():
    parser = argparse.ArgumentParser(description="Data Augmentor")
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()
    directory = args.path
    for filename in tqdm(os.listdir(directory),desc='Augmenting Data'):
        if filename.endswith(".jpg"):
            img = load_image(filename).astype(np.float32)
            bright = img + 30
            dark = img - 30
            horizontal_flip = img[:, ::-1]
            blured_image = cv2.GaussianBlur(img, (5,5),0)
            blured_image_dark = cv2.GaussianBlur(dark, (5,5),0)
            pre,post = filename.split('.')
            pre=pre+'_1'
            fname=pre+'.'+post
            cv2.imwrite(os.path.join(directory,fname),horizontal_flip)
            pre,post = filename.split('.')
            pre=pre+'_2'
            fname=pre+'.'+post
            cv2.imwrite(os.path.join(directory,fname),blured_image)
            pre,post = filename.split('.')
            pre=pre+'_3'
            fname=pre+'.'+post
            cv2.imwrite(os.path.join(directory,fname),bright)
            pre,post = filename.split('.')
            pre=pre+'_4'
            fname=pre+'.'+post
            cv2.imwrite(os.path.join(directory,fname),dark)
            pre,post = filename.split('.')
            pre=pre+'_5'
            fname=pre+'.'+post
            cv2.imwrite(os.path.join(directory,fname),blured_image_dark)
            
if __name__ == '__main__':
    main()
