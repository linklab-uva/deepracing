import torch
import data_loading.image_loading as il
import ml_models.PilotNet as PilotNet
im = il.load_image('test_img.jpg',size=(66,200), scale_factor=255.0)
print(im)
print(im.shape)
tensor = torch.from_numpy(im)
print(tensor)
print(tensor.shape)
pn = PilotNet()
print(pn)
