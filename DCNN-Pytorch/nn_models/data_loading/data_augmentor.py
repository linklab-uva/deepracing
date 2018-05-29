import numpy as np
import cv2
import argparse
import os
from tqdm import tqdm as tqdm
import xml.etree.ElementTree


def load_image(filepath):
    img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    return img

def main():
    parser = argparse.ArgumentParser(description="Data Augmentor")
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()
    directory = args.path
    image_directory = os.path.join(directory,"raw_images")
    annotation_directory = os.path.join(directory,"raw_annotations")
    for filename in tqdm(os.listdir(image_directory),desc='Augmenting Data'):
        if filename.endswith(".jpg"):
            xml_name = filename.split('_')
            xml_name = xml_name[2]
            xml_name = 'raw_data_point_'+xml_name
    
            img = load_image(os.path.join(image_directory,filename)).astype(np.float32)
            bright = img + 30
            dark = img - 30
            horizontal_flip = img[:, ::-1]
            blured_image = cv2.GaussianBlur(img, (5,5),0)
            blured_image_dark = cv2.GaussianBlur(dark, (5,5),0)
                        
            xml_name_globe = xml_name.split('.')[0] + '.xml'
            et = xml.etree.ElementTree.parse(os.path.join(annotation_directory,xml_name_globe)) 
            root = et.getroot()
            for steer in root[0].iter('m-steer'):
                if(float(steer.text)!=0):
                    new_steer = -float(steer.text)
                    steer.text = str(new_steer)
            
            et = xml.etree.ElementTree.parse(os.path.join(annotation_directory,xml_name_globe))
            xml_name = xml_name_globe.split('.')[0] + '_1.xml'
            et.write(os.path.join(annotation_directory,xml_name))
            pre,post = filename.split('.')
            pre=pre+'_1'
            fname=pre+'.'+post
            cv2.imwrite(os.path.join(image_directory,fname),horizontal_flip)
            
            et = xml.etree.ElementTree.parse(os.path.join(annotation_directory,xml_name_globe))
            xml_name = xml_name_globe.split('.')[0] + '_2.xml'
            et.write(os.path.join(annotation_directory,xml_name))
            pre,post = filename.split('.')
            pre=pre+'_2'
            fname=pre+'.'+post
            cv2.imwrite(os.path.join(image_directory,fname),blured_image)

            et = xml.etree.ElementTree.parse(os.path.join(annotation_directory,xml_name_globe))
            xml_name = xml_name_globe.split('.')[0] + '_3.xml'
            et.write(os.path.join(annotation_directory,xml_name))
            pre,post = filename.split('.')
            pre=pre+'_3'
            fname=pre+'.'+post
            cv2.imwrite(os.path.join(image_directory,fname),bright)

            et = xml.etree.ElementTree.parse(os.path.join(annotation_directory,xml_name_globe))
            xml_name = xml_name_globe.split('.')[0] + '_4.xml'
            et.write(os.path.join(annotation_directory,xml_name))
            pre,post = filename.split('.')
            pre=pre+'_4'
            fname=pre+'.'+post
            cv2.imwrite(os.path.join(image_directory,fname),dark)

            et = xml.etree.ElementTree.parse(os.path.join(annotation_directory,xml_name_globe))
            xml_name = xml_name_globe.split('.')[0] + '_5.xml'
            et.write(os.path.join(annotation_directory,xml_name))
            pre,post = filename.split('.')
            pre=pre+'_5'
            fname=pre+'.'+post
            cv2.imwrite(os.path.join(image_directory,fname),blured_image_dark)
            
            
if __name__ == '__main__':
    main()
