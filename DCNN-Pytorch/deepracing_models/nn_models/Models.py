import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import torch
import torch.nn.utils.rnn as RNNUtils
import sys
import deepracing_models.math_utils as mu
import math


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


        self.tanh = nn.Tanh()
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        # This flattens the output of the previous layer into a vector.
        out = out.view(out.size(0), -1) 
        out = self.fc1(out)
        out = self.tanh(out)

        out = self.fc2(out)
        out = self.tanh(out)

        out = self.fc3(out)
        out = torch.clamp(out,-1.0,1.0)

        out = self.prediction_layer(out)
        out = torch.clamp(out,-1.0,1.0)
        #out = out.unsqueeze(2)
        #print(out.size())
        return torch.clamp(out, -1.0, 1.0)

class CNNLSTM(nn.Module):
    def __init__(self, input_channels=3, output_dimension = 2, context_length=5, sequence_length=1, hidden_dimension = 100):
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
        self.hidden_dim = hidden_dimension
        self.context_length = context_length
        self.sequence_length = sequence_length
        
        self.rnn_init_hidden = torch.nn.Parameter(torch.normal(0, 1, size=(1,self.hidden_dim)), requires_grad=True)
        self.rnn_init_cell = torch.nn.Parameter(torch.normal(0, 1, size=(1,self.hidden_dim)), requires_grad=True)

        self.rnn = nn.LSTM(self.feature_length, self.hidden_dim, batch_first = True)
 
        # Linear layers.
        self.prediction_layer = nn.Linear(self.hidden_dim, self.output_dimension)

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

        return torch.clamp(predictions, -1.0, 1.0)

def generate3DConv(input_channels, relu, tanh):
    conv3d1 = nn.Conv3d(input_channels, 10, kernel_size=(5,3,3), stride = (1,2,2), padding=(2,0,0) )
    Norm3d_1 = nn.BatchNorm3d(10)
    conv3d2 = nn.Conv3d(10, 20, kernel_size=(5,3,3), stride = (1,2,2), padding=(2,0,0) )
    Norm3d_2 = nn.BatchNorm3d(20)
    conv3d3 = nn.Conv3d(20, 40, kernel_size=(3,3,3), stride = (1,2,2), padding=(1,0,0) )
    Norm3d_3 = nn.BatchNorm3d(40) 
    Pool3d_1 = torch.nn.MaxPool3d(3, stride=(1,1,1), padding=(1,0,0) )
    conv3d4 = nn.Conv3d(40, 120, kernel_size=(3,3,3), stride = (1,1,1), padding=(1,1,1) )
    Norm3d_4 = nn.BatchNorm3d(120) 
    conv3d5 = nn.Conv3d(120, 120, kernel_size=(3,3,3), stride = (1,1,1), padding=(1,1,1) )
    Norm3d_5 = nn.BatchNorm3d(120) 
    conv3d6 = nn.Conv3d(120, 240, kernel_size=(3,3,3), stride = (1,1,1), padding=(1,1,1) )
    Norm3d_6 = nn.BatchNorm3d(240) 
    Pool3d_2 = torch.nn.AvgPool3d(3, stride=(1,1,1), padding=(1,0,0))

    projection_encoder = torch.nn.Sequential(*[
        conv3d1,
        Norm3d_1,
        conv3d2,
        Norm3d_2,
        relu,
        conv3d3,
        Norm3d_3,
        relu,
        Pool3d_1,
        conv3d4,
        Norm3d_4,
        tanh,
        conv3d5,
        Norm3d_5,
        tanh,
        conv3d6,
        Norm3d_6,
        tanh,
        Pool3d_2,
    ])
    return projection_encoder
class AdmiralNetKinematicPredictor(nn.Module):
    def __init__(self, input_channels=3, output_dimension=2, sequence_length=10, \
                 context_length = 15, hidden_dim = 100, num_recurrent_layers = 1,  \
                     learnable_initial_state=True, use_3dconv=True):
        super(AdmiralNetKinematicPredictor, self).__init__()
        self.imsize = (66,200)
        self.input_channels = input_channels
        self.sequence_length = sequence_length
        self.context_length = context_length
        self.output_dimension = output_dimension
        self.use_3dconv = use_3dconv
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

        self.projection_features = 240*self.context_length * 3 * 20
        #projection encoder
        if self.use_3dconv:
            self.intermediate_projection_size = int(self.projection_features/self.sequence_length)
            self.projection_layer = nn.Linear(self.intermediate_projection_size, self.img_features)
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
        else:
            self.projection_feature_sequence = nn.Parameter(torch.normal(0,0.5, size=(self.sequence_length,self.img_features)), requires_grad=learnable_initial_state)

        #recurrent layers
        self.hidden_dim = hidden_dim
        self.linear_rnn = nn.LSTM(self.img_features, self.hidden_dim, batch_first = True, num_layers = num_recurrent_layers)
        self.linear_rnn_init_hidden = torch.nn.Parameter(torch.normal(0, 1, size=(1,self.hidden_dim)), requires_grad=learnable_initial_state)
        self.linear_rnn_init_cell = torch.nn.Parameter(torch.normal(0, 1, size=(1,self.hidden_dim)), requires_grad=learnable_initial_state)



        
    
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
        
        if self.use_3dconv:
            conv3d_out = self.projection_encoder( x.view(batch_size, self.input_channels, self.context_length, self.imsize[0], self.imsize[1]) )
            #print(conv3d_out.shape)
            projection_in = conv3d_out.view(batch_size, self.sequence_length, self.intermediate_projection_size)
            projection_features = self.projection_layer(projection_in)
        else:
            projection_features = self.projection_feature_sequence.unsqueeze(0).repeat(batch_size,1,1)

        x_linear, (final_hidden_position, final_cell_position) = self.linear_rnn(  projection_features , (linear_new_hidden, linear_new_cell) )

        position_predictions = self.classifier(x_linear)

        return position_predictions
        
class LinearRecursionCurvePredictor(nn.Module):
    def __init__(self, input_features, context_length = 5, hidden_dimension = 200, bezier_order=5,  output_dimension = 2):
        super(LinearRecursionCurvePredictor, self).__init__()
        self.linear_rnn = nn.LSTM(input_features, hidden_dimension, batch_first = True, num_layers = 1, bidirectional=False)
        self.linear_rnn_init_hidden = torch.nn.Parameter(torch.normal(0, 0.01, size=(self.linear_rnn.num_layers*(int(self.linear_rnn.bidirectional)+1),1,hidden_dimension)), requires_grad=True)
        self.linear_rnn_init_cell = torch.nn.Parameter(torch.normal(0, 0.01, size=(self.linear_rnn.num_layers*(int(self.linear_rnn.bidirectional)+1),1,hidden_dimension)), requires_grad=True)
        self.conv1 = nn.Conv2d(1, 24, kernel_size=3, padding=1)
        self.Norm_1 = nn.BatchNorm2d(24)
        self.conv2 = nn.Conv2d(24, 36, kernel_size=3, padding=1)
        self.Norm_2 = nn.BatchNorm2d(36)
        self.conv3 = nn.Conv2d(36, 36, kernel_size=3, padding=1)
        self.Norm_3 = nn.BatchNorm2d(36) 
        self.conv4 = nn.Conv2d(36, 48, kernel_size=3)
        self.Norm_4 = nn.BatchNorm2d(48)
        self.conv5 = nn.Conv2d(48, 64, kernel_size=3)
        self.Norm_5 = nn.BatchNorm2d(64)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.output_dimension = output_dimension
        self.bezier_order = bezier_order

        self.convolutions = torch.nn.Sequential(
            *[
            self.conv1,
            self.Norm_1,
            self.sigmoid,
            self.conv2,
            self.Norm_2,
            self.sigmoid,
            self.conv3,
            self.Norm_3,
            self.sigmoid,
            self.conv4,
            self.Norm_4,
            self.sigmoid,
            self.conv5,
            self.Norm_5
            ]
        )

        lineardimstart = 64*(hidden_dimension-4)
        self.linear_layers = torch.nn.Sequential(
            *[
                nn.Linear(lineardimstart,int(lineardimstart/2)),
                self.sigmoid,
                nn.Linear(int(lineardimstart/2),int(lineardimstart/4)),
                self.sigmoid,
                nn.Linear(int(lineardimstart/4),int(lineardimstart/8)),
                self.sigmoid,
                nn.Linear(int(lineardimstart/8),int(lineardimstart/16)),
                self.sigmoid,
                nn.Linear(int(lineardimstart/16),(bezier_order+1)*output_dimension),
            ]
        )


    def forward(self, inputs):
        batch_size = inputs.shape[0]
        linear_rnn_init_hidden = self.linear_rnn_init_hidden.repeat(1,batch_size,1)
        linear_rnn_init_cell = self.linear_rnn_init_cell.repeat(1,batch_size,1)
        recursive_features, (linear_new_hidden, linear_new_cell) = self.linear_rnn(inputs, (linear_rnn_init_hidden,  linear_rnn_init_cell) )
        recursive_features = recursive_features.unsqueeze(1)
        convout = self.convolutions(recursive_features)
        convflattened = convout.view(batch_size,-1)
        linearout = self.linear_layers(convflattened)
        return linearout.view(-1,self.bezier_order+1,self.output_dimension)

class AdmiralNetCurvePredictor(nn.Module):
    def __init__(self, input_channels=3, params_per_dimension=11, \
                 context_length = 5, hidden_dim = 200, num_recurrent_layers = 1, rnn_bidirectional=False,  \
                    additional_rnn_calls=25, learnable_initial_state=True, output_dimension = 2, use_3dconv=True):
        super(AdmiralNetCurvePredictor, self).__init__()
        self.imsize = (66,200)
        self.input_channels = input_channels
        self.params_per_dimension = params_per_dimension
        self.context_length = context_length
        self.num_recurrent_layers = num_recurrent_layers
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




        self.projection_features = 240*self.context_length * 3 * 20
        self.additional_rnn_calls = additional_rnn_calls
        self.intermediate_projection_size = int(self.projection_features/self.additional_rnn_calls)
        self.use_3dconv = use_3dconv
        if self.use_3dconv:
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
            self.projection_layer = nn.Linear(self.intermediate_projection_size, self.img_features)
        else:
            self.projection_features = torch.nn.Parameter(torch.normal(0, 0.01, size=(self.additional_rnn_calls, self.img_features), requires_grad=learnable_initial_state))


        #recurrent layers
        self.hidden_dim = hidden_dim
        self.linear_rnn = nn.LSTM(self.img_features, self.hidden_dim, batch_first = True, num_layers = num_recurrent_layers, bidirectional=rnn_bidirectional)
        self.linear_rnn_init_hidden = torch.nn.Parameter(torch.normal(0, 0.01, size=(self.linear_rnn.num_layers*(int(self.linear_rnn.bidirectional)+1),self.hidden_dim)), requires_grad=learnable_initial_state)
        self.linear_rnn_init_cell = torch.nn.Parameter(torch.normal(0, 0.01, size=(self.linear_rnn.num_layers*(int(self.linear_rnn.bidirectional)+1),self.hidden_dim)), requires_grad=learnable_initial_state)


        
    
        # Sub-convolutional layers.
        self.subConv1 = nn.Conv2d(1, 16, kernel_size=5, stride=(2,2), padding=(2,2))
        self.subConvNorm_1 = nn.BatchNorm2d(self.subConv1.out_channels)
        self.subConv2 = nn.Conv2d(16, 32, kernel_size=5, stride=(1,2), padding=(2,2))
        self.subConvNorm_2 = nn.BatchNorm2d(self.subConv2.out_channels)
        self.subConv3 = nn.Conv2d(32, 64, kernel_size=5, stride=1)
        self.subConvNorm_3 = nn.BatchNorm2d(self.subConv3.out_channels)
        self.subConvPool_1 = torch.nn.MaxPool2d(3, stride=(1,1))
        self.subConv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.subConvNorm_4 = nn.BatchNorm2d(self.subConv4.out_channels)
        self.subConv5= nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.subConvNorm_5 = nn.BatchNorm2d(self.subConv5.out_channels)
        self.subConvPool_2 = torch.nn.MaxPool2d(3, stride=(1,1))


        self.hidden_decoder = torch.nn.Sequential(*[
        self.subConv1,
        self.subConvNorm_1,
        self.relu,
        self.subConv2,
        self.subConvNorm_2,
        self.subConv3,
        self.relu,
        self.subConvNorm_3,
        self.relu,
        self.subConvPool_1,
        self.subConv4,
        self.subConvNorm_4,
        self.relu,
        self.subConv5,
        self.subConvNorm_5,
        self.relu,
        self.subConvPool_2,
        ])
        self.hidden_decoder_features = 2432
        self.classifier = nn.Sequential(*[
            nn.Linear(self.hidden_decoder_features, 1200),
            self.relu,
            nn.Linear(1200, 500),
            self.tanh,
            nn.Linear(500, self.params_per_dimension)
            #nn.Linear(2432, self.params_per_dimension)
            ]
        )
        

    def forward(self, x):
        #resize for convolutional layers
        batch_size = x.shape[0] 
        #print(y.shape)
        convin = x.view(-1, self.input_channels, self.imsize[0], self.imsize[1]) 
        convout = self.state_encoder(convin)
        context_in = convout.view(batch_size , self.context_length , self.img_features)

        linear_rnn_init_hidden = self.linear_rnn_init_hidden.unsqueeze(1).repeat(1,batch_size,1)
        linear_rnn_init_cell = self.linear_rnn_init_cell.unsqueeze(1).repeat(1,batch_size,1)

        # linear_rnn_init_hidden = self.linear_rnn_init_hidden.expand(self.linear_rnn_init_hidden.shape[0],batch_size,self.linear_rnn_init_hidden.shape[1])
        # linear_rnn_init_cell = self.linear_rnn_init_cell.expand(self.linear_rnn_init_cell.shape[0],batch_size,self.linear_rnn_init_cell.shape[1])
        #print(context_in.shape)
        # = RNNUtils.pack_padded_sequence(context_in, (context_in.shape[1]*np.ones(context_in.shape[0])).tolist() , batch_first=True, enforce_sorted=False)
        _, (linear_new_hidden, linear_new_cell) = self.linear_rnn(context_in, (linear_rnn_init_hidden,  linear_rnn_init_cell) )
        
    
        #print(conv3d_out.shape)
        if self.use_3dconv:
            conv3d_out = self.projection_encoder( x.view(batch_size, self.input_channels, self.context_length, self.imsize[0], self.imsize[1]) )
            projection_in = conv3d_out.view(batch_size, self.additional_rnn_calls, self.intermediate_projection_size)
            projection_features = self.projection_layer(projection_in)
        else:
            projection_features = self.projection_features.expand(batch_size,self.projection_features.shape[0],self.projection_features.shape[1])
        x_linear, (final_hidden_position, final_cell_position) = self.linear_rnn(  projection_features , (linear_new_hidden, linear_new_cell) )
        x_linear_unsqueeze = x_linear.unsqueeze(1)
        hidden_convout = self.hidden_decoder(x_linear_unsqueeze)
        x_features = hidden_convout.view(batch_size,self.output_dimension,self.hidden_decoder_features)

        return self.classifier(x_features)
class AdmiralNetCombinedBezierPredictor(nn.Module):
    def __init__(self, params_per_dimension=8, context_length = 5, hidden_dim = 200, \
                num_recurrent_layers = 1, rnn_bidirectional=False, additional_rnn_calls=25, learnable_initial_state=True, output_dimension = 2):
        super(AdmiralNetCombinedBezierPredictor, self).__init__()
        self.pos_predictor = AdmiralNetCurvePredictor(input_channels = 3, params_per_dimension=params_per_dimension, context_length = context_length, hidden_dim = hidden_dim, \
                num_recurrent_layers = num_recurrent_layers, rnn_bidirectional=rnn_bidirectional, \
                     additional_rnn_calls=additional_rnn_calls, learnable_initial_state=learnable_initial_state, output_dimension = output_dimension)
        self.vel_predictor = AdmiralNetCurvePredictor(input_channels = 2, params_per_dimension=params_per_dimension, context_length = context_length, hidden_dim = hidden_dim, \
                num_recurrent_layers = num_recurrent_layers, rnn_bidirectional=rnn_bidirectional, \
                     additional_rnn_calls=additional_rnn_calls, learnable_initial_state=learnable_initial_state, output_dimension = output_dimension)
    def forward(self, images, opt_flow):
        pos_predictions = self.pos_predictor(images)
        vel_predictions = self.vel_predictor(opt_flow)
        return pos_predictions, vel_predictions

