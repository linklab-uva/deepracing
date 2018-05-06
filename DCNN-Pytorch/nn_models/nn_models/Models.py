import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import numpy as np
class PilotNet(nn.Module):
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
        self.Norm_1 = nn.BatchNorm2d(24)
        self.Norm_2 = nn.BatchNorm2d(36)
        self.Norm_3 = nn.BatchNorm2d(48) 
        self.Norm_4 = nn.BatchNorm2d(64)
        # Linear layers.
        self.fc1 = nn.Linear(64*1*18, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.prediction_layer = nn.Linear(10, self.output_size)

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.Norm_1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.Norm_2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.Norm_3(out)
        out = self.relu(out)
        out = self.conv4(out)
        out = self.Norm_4(out)
        out = self.relu(out)
        out = self.conv5(out)
        out = self.relu(out)
        # This flattens the output of the previous layer into a vector.
        out = out.view(out.size(0), -1) 
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        predictions = self.prediction_layer(out)
        return predictions

class AdmiralNet(nn.Module):
    """PyTorch Implementation of NVIDIA's PilotNet"""
    def __init__(self, seq_length=25):
        super(AdmiralNet, self).__init__()
        # Convolutional layers.
        self.output_size = 1
        self.conv1 = nn.Conv2d(3, 24, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(36, 48, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3)
        #batch norm layers
        self.Norm_1 = nn.BatchNorm2d(24)
        self.Norm_2 = nn.BatchNorm2d(36)
        self.Norm_3 = nn.BatchNorm2d(48) 
        self.Norm_4 = nn.BatchNorm2d(64)
        
        #recurrent layers
        self.seq_length=seq_length
        self.lstm = nn.LSTM(64*1*18, 100, batch_first = True)

        # Linear layers.
        self.prediction_layer = nn.Linear(100, self.output_size)

        #activations
        self.relu = nn.ReLU()

    def forward(self, x):
        #resize for convolutional layers
        x = x.view(-1, 3, 66, 200) 
        x = self.conv1(x)
        x = self.Norm_1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.Norm_2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.Norm_3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.Norm_4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.relu(x)
        #print(x.shape)
        # Unpack for the LSTM.
        x = x.view(-1, self.seq_length, 64*1*18) 
        x, finalHiddenState = self.lstm(x)
        predictions = self.prediction_layer(x)
        return predictions

