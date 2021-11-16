import torchvision, torchvision.transforms.functional as F, torch
from PIL.ImageFilter import GaussianBlur as GBPIL
import PIL, PIL.Image
class IdentityTransform(torchvision.transforms.Lambda):
    '''
    Does nothing, just returns the provided image as-is
    
    ''' 
    def __init__(self):
        super(IdentityTransform, self).__init__(lambda image: image)
class AddGaussianNoise(object):
    '''
    Optionally add some white noise to an image.
    ''' 
    def __init__(self, stdev):
        self.stdev = stdev
    def __call__(self, img):
        if self.stdev<=0.0:
            return img
        if isinstance(img, PIL.Image.Image):
            imtensor = F.to_tensor(img)
        elif isinstance(img, torch.Tensor):
            imtensor = img
        else:
            raise TypeError("This transform only applies to PIL images and Torch tensors. Got unknown type: %s" % (str(type(img)),))
        return F.to_pil_image(torch.clamp(imtensor + self.stdev*torch.randn_like(imtensor), 0.0, 1.0))
    def __repr__(self):
        return self.__class__.__name__ + '()'
class GaussianBlur(object):
    '''
    Calls PIL's GaussianBlur filter with the specified radius
    ''' 
    def __init__(self, radius):
        #super(IdentityTransform, self).__init__(lambda image: image)
        self.gbPIL = GBPIL(radius=radius)
    def __call__(self, img):
        return img.filter(self.gbPIL)
    def __repr__(self):
        return self.__class__.__name__ + '()'