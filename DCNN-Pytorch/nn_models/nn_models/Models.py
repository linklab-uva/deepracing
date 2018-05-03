import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
class PilotNet(nn.Module):
    '''
        Caffe2 NetDef we are mimicking.
    	 # Image size: 66x200 -> 31x98
        self.conv1 = brew.conv(model, input_blob, 'conv1', dim_in=self.input_channels, dim_out=24, kernel=5, stride=2)
	    # Image size: 31x98 -> 14x47
        self.conv2 = brew.conv(model, self.conv1, 'conv2', dim_in=24, dim_out=36, kernel=5, stride=2)
	    # Image size: 14x47 -> 5x22
        self.conv3 = brew.conv(model, self.conv2, 'conv3', dim_in=36, dim_out=48, kernel=5, stride=2)
	    # Image size: 5x22 -> 3x20
        self.conv4 = brew.conv(model, self.conv3, 'conv4', dim_in=48, dim_out=64, kernel=3)
	    # Image size: 3x20 -> 1x18
        self.conv5 = brew.conv(model, self.conv4, 'conv5', dim_in=64, dim_out=64, kernel=3)
	    # Flatten from 64 X 1 X 18 image to the "deep feature" vector
        self.deep_features = model.net.Reshape("conv5", ["deep_features", "conv5_old"], shape=[-1, 64*1*18])
        self.fc1 = brew.fc(model, 'deep_features', 'fc1', dim_in=64*1*18, dim_out=100, axis=1)
        self.fc2 = brew.fc(model, self.fc1, 'fc2', dim_in=100, dim_out=50, axis=1)
        self.fc3 = brew.fc(model, self.fc2, 'fc3', dim_in=50, dim_out=10, axis=1)
        self.prediction = brew.fc(model, self.fc3, 'prediction', dim_in=10, dim_out=self.output_dim, axis=1)
        # Create a copy of the current net. We will use it on the forward
    '''
    """PyTorch Implementation of NVIDIA's PilotNet"""
    def __init__(self):
        super(PilotNet, self).__init__()
        # Convolutional layers.
        self.output_size = 1
        self.conv1 = nn.Conv2d(3, 24, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(36, 48, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3)
        
        # Linear layers.
        self.fc1 = nn.Linear(64*1*18, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.prediction_layer = nn.Linear(10, self.output_size)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        # This flattens the output of the previous layer into a vector.
        out = out.view(out.size(0), -1) 
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        predictions = self.prediction_layer(out)
        return predictions



