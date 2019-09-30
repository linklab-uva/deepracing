import torchvision
class IdentifyTransform(torchvision.transforms.Lambda):
    '''
    Does nothing, just returns the provided image as-is
    
    ''' 
    def __init__(self):
        super(IdentifyTransform, self).__init__(lambda image: image)