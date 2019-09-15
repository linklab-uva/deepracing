import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import torch
import torchvision.models as visionmodels
import torchvision.models.vgg
import sys
class ResNetAdapter(nn.Module):
    def __init__(self, pretrained = True):
        super(ResNetAdapter, self).__init__()
        resnet_model = visionmodels.resnet152(pretrained = pretrained)
        self.activation = nn.Tanh()
        self.features = nn.Sequential(*list(resnet_model.children())[:-2])
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.features(x)
        x = x.view(batch_size, -1)
        return x

class PilotNet(nn.Module):
    """PyTorch Implementation of NVIDIA's PilotNet"""
    def __init__(self, input_channels = 3, output_dim = 1):
        super(PilotNet, self).__init__()
        # Convolutional layers.
        self.output_size = output_dim
        self.conv1 = nn.Conv2d(input_channels, 24, kernel_size=5, stride=2)
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
        #out = out.unsqueeze(2)
        #print(out.size())
        return out

class CNNLSTM(nn.Module):
    def __init__(self, input_channels=3, output_dimension = 3, context_length=5, sequence_length=1, hidden_dim = 100):
        super(CNNLSTM, self).__init__()
        #self.input_channels = 5
        self.input_channels = input_channels
        # Convolutional layers.

        self.output_dimension = output_dimension
        self.conv1 = nn.Conv2d(self.input_channels, 24, kernel_size=5, stride=2)
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
        self.img_features = 1*64*18
        self.feature_length = (1*64*18)
        self.hidden_dim = hidden_dim
        self.context_length = context_length
        self.sequence_length = sequence_length
        
        self.rnn_init_hidden = torch.nn.Parameter(torch.normal(0, 1, size=(1,self.hidden_dim)), requires_grad=True)
        self.rnn_init_cell = torch.nn.Parameter(torch.normal(0, 1, size=(1,self.hidden_dim)), requires_grad=True)

        self.rnn = nn.LSTM(self.feature_length, hidden_dim, batch_first = True)
 
        # Linear layers.
        self.prediction_layer = nn.Linear(hidden_dim, self.output_size)

        #activations
        self.relu = nn.ReLU()

        self.projector_input = torch.nn.Parameter(torch.normal(0, 1, size=(self.sequence_length, self.feature_length)), requires_grad=True)
        
               
    def forward(self, x):
        #resize for convolutional layers
        batch_size = x.shape[0]
      #  print(x.shape)
        x1 = x.view(-1, self.input_channels, 66, 200) 
        #print(x1.shape)
        x2 = self.conv1(x1)
        x3 = self.Norm_1(x2)
        x4 = self.relu(x3)
        x5 = self.conv2(x4)
        x6 = self.Norm_2(x5)
        x7 = self.relu(x6)
        x8 = self.conv3(x7)
        x9 = self.Norm_3(x8)
        x10 = self.relu(x9)
        x11 = self.conv4(x10)
        x12 = self.Norm_4(x11)
        x13 = self.relu(x12)
        x14 = self.conv5(x13)
        x15 = self.relu(x14)
        #maps=[x1,x2,x3,x4,x5]
        # Unpack for the RNN.
       # print(x15.shape)
        x16 = x15.view(batch_size, self.context_length, self.img_features)

        rnn_init_hidden = self.rnn_init_hidden.repeat(1,batch_size,1)
        rnn_init_cell = self.rnn_init_cell.repeat(1,batch_size,1)
        _, (new_hidden, new_cell) = self.rnn(x16, (rnn_init_hidden,  rnn_init_cell) )     
     #   print(new_hidden[0].shape)   
      #  print(init_hidden[1].shape)
        
        projector = self.projector_input.repeat(batch_size,1,1)
        x17, final_hidden = self.rnn( projector, (new_hidden, new_cell) )

        predictions = self.prediction_layer(x17)

        return predictions


class AdmiralNetKinematicPredictor(nn.Module):
    def __init__(self, input_channels=3, output_dimension=3, sequence_length=10, \
                 context_length = 15, hidden_dim = 100, num_recurrent_layers = 1,  \
                     learnable_initial_state=True):
        super(AdmiralNetKinematicPredictor, self).__init__()
        self.imsize = (66,200)
        self.input_channels = input_channels
        self.sequence_length = sequence_length
        self.context_length = context_length
        self.output_dimension = output_dimension
        #activations
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        # Convolutional layers.
        self.conv1 = nn.Conv2d(self.input_channels, 24, kernel_size=5, stride=2)
        self.Norm_1 = nn.BatchNorm2d(24)
        self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2)
        self.Norm_2 = nn.BatchNorm2d(36)
        self.conv3 = nn.Conv2d(36, 48, kernel_size=5, stride=2)
        self.Norm_3 = nn.BatchNorm2d(48) 
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3)
        self.Norm_4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3)
        self.Norm_5 = nn.BatchNorm2d(64)

        self.state_encoder = torch.nn.Sequential(*[
        self.conv1,
        self.Norm_1,
        self.conv2,
        self.Norm_2,
        self.conv3,
        self.Norm_3,
        self.conv4,
        self.Norm_4,
        self.conv5,
        self.Norm_5
        ])
        self.img_features = 1*64*18

        #projection encoder
        self.conv3d1 = nn.Conv3d(input_channels, 10, kernel_size=(5,3,3), stride = (1,2,2), padding=(2,0,0) )
        self.Norm3d_1 = nn.BatchNorm3d(10)
        self.conv3d2 = nn.Conv3d(10, 20, kernel_size=(5,3,3), stride = (1,2,2), padding=(2,0,0) )
        self.Norm3d_2 = nn.BatchNorm3d(20)
        self.conv3d3 = nn.Conv3d(20, 40, kernel_size=(3,3,3), stride = (1,2,2), padding=(1,0,0) )
        self.Norm3d_3 = nn.BatchNorm3d(40) 
        self.Pool3d_1 = torch.nn.MaxPool3d(3, stride=(1,1,1), padding=(1,0,0) )
        self.conv3d4 = nn.Conv3d(40, 120, kernel_size=(3,3,3), stride = (1,1,1), padding=(1,1,1) )
        self.Norm3d_4 = nn.BatchNorm3d(120) 
        self.conv3d5 = nn.Conv3d(120, 120, kernel_size=(3,3,3), stride = (1,1,1), padding=(1,1,1) )
        self.Norm3d_5 = nn.BatchNorm3d(120) 
        self.conv3d6 = nn.Conv3d(120, 240, kernel_size=(3,3,3), stride = (1,1,1), padding=(1,1,1) )
        self.Norm3d_6 = nn.BatchNorm3d(240) 
        self.Pool3d_2 = torch.nn.AvgPool3d(3, stride=(1,1,1), padding=(1,0,0))

        self.projection_encoder = torch.nn.Sequential(*[
            self.conv3d1,
            self.Norm3d_1,
            self.conv3d2,
            self.Norm3d_2,
            self.relu,
            self.conv3d3,
            self.Norm3d_3,
            self.relu,
            self.Pool3d_1,
            self.conv3d4,
            self.Norm3d_4,
            self.tanh,
            self.conv3d5,
            self.Norm3d_5,
            self.tanh,
            self.conv3d6,
            self.Norm3d_6,
            self.tanh,
            self.Pool3d_2,
        ])


        #recurrent layers
        self.hidden_dim = hidden_dim
        self.linear_rnn = nn.LSTM(self.img_features, self.hidden_dim, batch_first = True, num_layers = num_recurrent_layers)
        self.linear_rnn_init_hidden = torch.nn.Parameter(torch.normal(0, 1, size=(1,self.hidden_dim)), requires_grad=learnable_initial_state)
        self.linear_rnn_init_cell = torch.nn.Parameter(torch.normal(0, 1, size=(1,self.hidden_dim)), requires_grad=learnable_initial_state)


        self.projection_features = 240*self.context_length * 3 * 20
        self.intermediate_projection_size = int(self.projection_features/self.sequence_length)
        self.projection_layer = nn.Linear(self.intermediate_projection_size, self.img_features)

        
    
        # Linear layers.
        self.classifier = nn.Linear(self.hidden_dim, self.output_dimension)
        

    def forward(self, x):
        #resize for convolutional layers
        batch_size = x.shape[0] 
        #print(y.shape)
        convin = x.view(-1, self.input_channels, self.imsize[0], self.imsize[1]) 
        convout = self.state_encoder(convin)
        context_in = convout.view(batch_size , self.context_length , self.img_features)

        linear_rnn_init_hidden = self.linear_rnn_init_hidden.unsqueeze(1).repeat(1,batch_size,1)
        linear_rnn_init_cell = self.linear_rnn_init_cell.unsqueeze(1).repeat(1,batch_size,1)
        _, (linear_new_hidden, linear_new_cell) = self.linear_rnn(context_in, (linear_rnn_init_hidden,  linear_rnn_init_cell) )
        
      
        conv3d_out = self.projection_encoder( x.view(batch_size, self.input_channels, self.context_length, self.imsize[0], self.imsize[1]) )
        #print(conv3d_out.shape)
        projection_in = conv3d_out.view(batch_size, self.sequence_length, self.intermediate_projection_size)
        projection_features = self.projection_layer(projection_in)

        x_linear, (final_hidden_position, final_cell_position) = self.linear_rnn(  projection_features , (linear_new_hidden, linear_new_cell) )

        position_predictions = self.classifier(x_linear)

        return position_predictions
