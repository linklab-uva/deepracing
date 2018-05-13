import numpy as np
import cv2
def load_image(filepath):
    img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    return img