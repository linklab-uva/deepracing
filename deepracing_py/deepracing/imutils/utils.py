import cv2
import skimage
import PIL
def readImage(filepath):
   # return skimage.util.img_as_ubyte(skimage.io.imread(filepath))
    return cv2.cvtColor(cv2.imread(filepath,flags=cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
def resizeImage(image, dsize, interpolation = cv2.INTER_AREA):
    if image.shape[0]==dsize[0] and image.shape[1]==dsize[1]:
        return image
    return cv2.resize(image,(dsize[1], dsize[0]), interpolation = interpolation)
def resizeImageFactor(image, factor, interpolation = cv2.INTER_AREA):
    if factor==1.0:
        return image
    return cv2.resize(image,None, fx=factor, fy=factor, interpolation = interpolation)
    