import numpy as np
import cv2
import argparse
import os

def load_image(filepath):
    img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    return img

def main():
    parser = argparse.ArgumentParser(description="Data Augmentor")
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()
    directory = args.path
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            img = load_image(filename).astype(np.float32)
            horizontal_flip = img[:, ::-1]
            blured_image = cv2.GaussianBlur(img, (5,5),0)
            pre,post = filename.split('.')
            pre=pre+'_1'
            filename=pre+'.'+post
            cv2.imwrite(os.path.join(directory,filename),horizontal_flip)
            pre,post = filename.split('.')
            pre=pre+'_2'
            filename=pre+'.'+post
            cv2.imwrite(os.path.join(directory,filename),blured_image)
            
if __name__ == '__main__':
    main()
