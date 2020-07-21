import torchvision
from PIL.ImageFilter import GaussianBlur as GBPIL
class IdentifyTransform(torchvision.transforms.Lambda):
    '''
    Does nothing, just returns the provided image as-is
    
    ''' 
    def __init__(self):
        super(IdentifyTransform, self).__init__(lambda image: image)
class GaussianBlur(object):
    '''
    Calls PIL's GaussianBlur filter with the specified radius
    ''' 
    def __init__(self, radius):
        #super(IdentifyTransform, self).__init__(lambda image: image)
        self.gbPIL = GBPIL(radius=radius)
    def __call__(self, img):
        return self.gbPIL.filter(img)
    def __repr__(self):
        return self.__class__.__name__ + '()'