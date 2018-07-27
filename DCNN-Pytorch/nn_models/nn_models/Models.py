import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import torch
import torchvision.models as visionmodels
import torchvision.models.vgg

class ResNetAdapter(nn.Module):
    def __init__(self):
        super(ResNetAdapter, self).__init__()
        resnet_model = visionmodels.resnet152(pretrained=True)
        self.features = nn.Sequential(*list(resnet_model.children())[:-2])
        self.classifier = nn.Sequential(*[nn.Linear(43008, 2048),\
                        nn.Linear(2048, 1024),\
                        nn.Linear(1024, 128),\
                        nn.Linear(128, 1)])
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        predictions = self.classifier(x)
        return predictions

class EnsignNet(nn.Module):
    """PyTorch Implementation of NVIDIA's PilotNet"""
    def __init__(self, sequence_length = 1):
        super(EnsignNet, self).__init__()
        # Convolutional layers.
        self.output_size = sequence_length
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
        out = self.prediction_layer(out)
        out = out.unsqueeze(2)
        return out

class PilotNet(nn.Module):
    """Upgraded Version of NVIDIA's PilotNet"""
    def __init__(self, sequence_length = 1):
        super(PilotNet, self).__init__()
        # Convolutional layers.
        self.output_size = sequence_length
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
        out = self.prediction_layer(out)
        out = out.unsqueeze(2)
        return out
class CommandantNet(nn.Module):
    def __init__(self, sequence_length=25, context_length = 25, hidden_dim = 100, use_float32 = False, gpu = -1, optical_flow = False):
        super(CommandantNet, self).__init__()
        self.gpu=gpu
        self.use_float32=use_float32
        # Convolutional layers.
        self.output_size = 1
        if optical_flow:
            self.input_channels = 2
        else:
            self.input_channels = 3
        self.conv1 = nn.Conv2d(self.input_channels, 12, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(12, 24, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(24, 36, kernel_size=3)
        self.pool1 = nn.MaxPool2d(3,3)
        self.conv4 = nn.Conv2d(36, 36, kernel_size=3)
        self.conv5 = nn.Conv2d(36, 24, kernel_size=3)
        self.conv6 = nn.Conv2d(24, 12, kernel_size=3)
        self.pool2 = nn.MaxPool2d(2,2)
        #batch norm layers
        self.Norm_1 = nn.BatchNorm2d(12)
        self.Norm_2 = nn.BatchNorm2d(24)
        self.Norm_3 = nn.BatchNorm2d(36) 
        self.Norm_4 = nn.BatchNorm2d(36)
        self.Norm_5 = nn.BatchNorm2d(24)
        
        #recurrent layers
        self.hidden_dim = hidden_dim
        self.sequence_length=sequence_length
        self.context_length = context_length
        self.feature_length = 157
        
        self.lstm = nn.LSTM(self.feature_length, hidden_dim, batch_first = True)

        # Linear layers.
        self.prediction_layer = nn.Linear(hidden_dim, self.output_size)
    
        #activations
        self.relu = nn.ReLU()

    def forward(self, x, previous_control):
        #resize for convolutional layers
        batch_size = x.shape[0]
        x = x.view(-1, self.input_channels, 125,400) 
        x = self.conv1(x)
        x = self.Norm_1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.Norm_2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.Norm_3(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.conv4(x)
        x = self.Norm_4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.Norm_5(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.relu(x)
        x = self.pool2(x)

        # Unpack for the LSTM.
        x = x.view(batch_size, self.context_length, self.feature_length-1) 
        x = torch.cat((x,previous_control),2)
        x, init_hidden = self.lstm(x) 
        if(self.use_float32):
            zeros = torch.zeros([batch_size, self.sequence_length, self.feature_length], dtype=torch.float32)
                
        else:
            zeros = torch.zeros([batch_size, self.sequence_length, self.feature_length], dtype=torch.float64)
        if(self.gpu>=0):
            zeros = zeros.cuda(self.gpu)
        x, final_hidden = self.lstm(zeros, init_hidden)
        predictions = self.prediction_layer(x)
        return predictions
class AdmiralNet(nn.Module):
    def __init__(self, cell='lstm', sequence_length=25, context_length = 25, hidden_dim = 100, use_float32 = False, gpu = -1, optical_flow = False):
        super(AdmiralNet, self).__init__()
        self.gpu=gpu
        self.use_float32=use_float32
        if optical_flow:
            self.input_channels = 2
        else:
            self.input_channels = 3
        # Convolutional layers.
        self.output_size = 1
        self.feats = nn.Sequential()
        self.feats.add_module("conv1",nn.Conv2d(self.input_channels, 24, kernel_size=5, stride=2))
        self.feats.add_module("Norm_1",nn.BatchNorm2d(24))
        self.feats.add_module("ReLU",nn.ReLU(True))
        self.feats.add_module("conv2",nn.Conv2d(24, 36, kernel_size=5, stride=2))
        self.feats.add_module("Norm_2",nn.BatchNorm2d(36))
        self.feats.add_module("ReLU2",nn.ReLU(True))
        self.feats.add_module("conv3",nn.Conv2d(36, 48, kernel_size=5, stride=2))
        self.feats.add_module("Norm_3",nn.BatchNorm2d(48))
        self.feats.add_module("ReLU3",nn.ReLU(True))
        self.feats.add_module("conv4",nn.Conv2d(48, 64, kernel_size=3))
        self.feats.add_module("Norm_4",nn.BatchNorm2d(64))
        self.feats.add_module("ReLU4",nn.ReLU(True))
        self.feats.add_module("conv5",nn.Conv2d(64, 64, kernel_size=3))
        self.feats.add_module("ReLU5",nn.ReLU(True))

        #recurrent layers
        self.img_features = 1*64*18
        self.feature_length = (1*64*18)
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.context_length = context_length
        self.cell = cell
        
        if(self.cell=='lstm'):
            self.rnn = nn.LSTM(self.feature_length, hidden_dim, batch_first = True)
        elif(self.cell=='gru'):
            self.rnn = nn.GRU(self.feature_length, hidden_dim, batch_first = True)
        else:
            self.rnn = nn.RNN(self.feature_length, hidden_dim, batch_first = True) 

        # Linear layers.
        self.prediction_layer = nn.Linear(hidden_dim, self.output_size)

    def forward(self, x, throttle, brake):
        batch_size = x.shape[0]
        #resize for convolutional layers
        x = x.view(-1, self.input_channels, 66, 200)
        #print(x.size())
        x = self.feats(x)
        
        #maps=[x1,x2,x3,x4,x5]
        # Unpack for the RNN.
        #print(x.size())
        #print(batch_size,self.context_length,self.img_features)
        x = x.view(batch_size, self.context_length, self.img_features)
        #throttle = throttle.view(throttle.shape[0],throttle.shape[1],-1)
        #brake = brake.view(brake.shape[0],brake.shape[1],-1)  
        #x = torch.cat((x,throttle),2)
        #x = torch.cat((x,brake),2)
        x, init_hidden = self.rnn(x) 
        if(self.use_float32):
            zeros = torch.zeros([batch_size, self.sequence_length, self.feature_length], dtype=torch.float32)
                
        else:
            zeros = torch.zeros([batch_size, self.sequence_length, self.feature_length], dtype=torch.float64)
        if(self.gpu>=0):
            zeros = zeros.cuda(self.gpu)
        x, final_hidden = self.rnn(zeros, init_hidden)
        predictions = self.prediction_layer(x)
        return predictions

class AdmiralNet_v2(nn.Module):
    def __init__(self,cell='lstm', sequence_length=25, context_length = 25, hidden_dim = 256, use_float32 = False, gpu = -1, optical_flow = False):
        super(AdmiralNet_v2, self).__init__()
        self.gpu=gpu
        self.use_float32=use_float32
        if optical_flow:
            self.input_channels = 2
        else:
            self.input_channels = 3
        # Convolutional layers.
        self.output_size = 1
        
        #RESIDUAL BLOCK 1
        self.conv1 = nn.Conv2d(self.input_channels, 24, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(24, 24, kernel_size=3, padding=1)
        self.conv1_3 = nn.Conv2d(24, 24, kernel_size=3, stride=2)
        
        #RESIDUAL BLOCK 2
        self.conv2 = nn.Conv2d(24, 36, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(36, 36, kernel_size=3, padding=1)
        self.conv2_3 = nn.Conv2d(36, 36, kernel_size=3, stride=2)
        
        #RESIDUAL BLOCK 3
        self.conv3 = nn.Conv2d(36, 48, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(48, 48, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(48, 48, kernel_size=3, stride=2)
        
        #RESIDUAL BLOCK 4
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3,padding=1)
        self.conv4_2 = nn.Conv2d(64, 64, kernel_size=3,padding=1)
        self.conv4_3 = nn.Conv2d(64, 64, kernel_size=3)
        
        #RESIDUAL BLOCK 5
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3,padding=1)
        self.conv5_2 = nn.Conv2d(64, 64, kernel_size=3,padding=1)
        self.conv5_3 = nn.Conv2d(64, 64, kernel_size=3)
        
        #batch norm layers
        self.Norm_1 = nn.BatchNorm2d(24)
        self.Norm_2 = nn.BatchNorm2d(36)
        self.Norm_3 = nn.BatchNorm2d(48) 
        self.Norm_4 = nn.BatchNorm2d(64)
        
        #recurrent layers
        self.feature_length = 1*64*60
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.context_length = context_length

        self.cell = cell
        
        #RNN support added
        if(self.cell=='lstm'):
            self.rnn = nn.LSTM(self.feature_length, hidden_dim, batch_first = True)
        elif(self.cell=='gru'):
            self.rnn = nn.GRU(self.feature_length, hidden_dim, batch_first = True)
        else:
            self.rnn = nn.RNN(self.feature_length, hidden_dim, batch_first = True) 

        # Linear layers.
        self.prediction_layer = nn.Linear(hidden_dim, self.output_size)
    
        #activations
        self.relu = nn.ReLU()

    def forward(self, x):
        #resize for convolutional layers
        batch_size = x.shape[0]
        x = x.view(-1, self.input_channels, 66, 200) 
        
        x = self.conv1(x)
        x = self.Norm_1(x)
        x1 = self.relu(x)
        x = self.conv1_2(x1)
        x = self.Norm_1(x)
        x = self.relu(x)
        x = self.conv1_3(x+x1)
        x = self.Norm_1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.Norm_2(x)
        x2 = self.relu(x)
        x = self.conv2_2(x2)
        x = self.Norm_2(x)
        x = self.relu(x)
        x = self.conv2_3(x+x2)
        x = self.Norm_2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.Norm_3(x)
        x3 = self.relu(x)
        x = self.conv3_2(x3)
        x = self.Norm_3(x)
        x = self.relu(x)
        x = self.conv3_3(x+x3)
        x = self.Norm_3(x)
        x = self.relu(x)
        
        x = self.conv4(x)
        x = self.Norm_4(x)
        x4 = self.relu(x)
        x = self.conv4_2(x4)
        x = self.Norm_4(x)
        x = self.relu(x)
        x = self.conv4_3(x+x4)
        x = self.Norm_4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.Norm_4(x)
        x5 = self.relu(x)
        x = self.conv5_2(x5)
        x = self.Norm_4(x)
        x = self.relu(x)
        x = self.conv5_3(x+x5)
        x = self.relu(x)

        #print(x.shape)
        # Unpack for the RNN.
        x = x.view(batch_size, self.context_length, self.feature_length) 
        x, init_hidden = self.rnn(x) 
        if(self.use_float32):
            zeros = torch.zeros([batch_size, self.sequence_length, self.feature_length], dtype=torch.float32)
                
        else:
            zeros = torch.zeros([batch_size, self.sequence_length, self.feature_length], dtype=torch.float64)
        if(self.gpu>=0):
            zeros = zeros.cuda(self.gpu)
        x, final_hidden = self.rnn(zeros, init_hidden)
        predictions = self.prediction_layer(x)
        return predictions

