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

class ConvolutionalAutoencoder(nn.Module):
    def __init__(self, out_size, in_channels):
        super(ConvolutionalAutoencoder, self).__init__()
        self.out_size = out_size
        self.elu = torch.nn.ELU()
        self.sigmoid = torch.nn.Sigmoid()
        self.conv_layers = nn.Sequential(*[
            nn.Conv2d(in_channels, 32, kernel_size = 5, stride=1, bias = False),
            nn.BatchNorm2d(32),
            self.elu,
            nn.Conv2d(32, 64, kernel_size = 5, stride=1, bias = False),
            nn.BatchNorm2d(64),
            self.elu,
            nn.Conv2d(64, 64, kernel_size = 5, bias = False),
            nn.BatchNorm2d(64),
            self.elu,
            nn.Conv2d(64, 64, kernel_size = 5, bias = False),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, out_size, kernel_size = 12, bias = True),
            self.elu,
        ])

        self.deconv_layers = nn.Sequential(*[
            nn.ConvTranspose2d(out_size, 64, kernel_size = 12, bias = True),
            nn.BatchNorm2d(64),
            self.elu,
            nn.ConvTranspose2d(64, 32, kernel_size = 5, bias = False),
            nn.BatchNorm2d(32),
            self.elu,
            nn.ConvTranspose2d(32, 16, kernel_size = 5, stride=1, bias = False),
            nn.BatchNorm2d(16),
            self.elu,
            nn.ConvTranspose2d(16, 16, kernel_size = 5, bias = False),
            nn.BatchNorm2d(16),
            self.elu,
            nn.ConvTranspose2d(16, in_channels, kernel_size = 5, stride=1, bias = False),
            self.sigmoid,
        ])




    def encoder(self, x):
        return self.conv_layers(x)

    def decoder(self, z):
        return self.deconv_layers(z)
     
    def forward(self, x):
        z = self.encoder(x)
        #print(z.shape)
        y = self.decoder(z)
        return z, y


class VariationalImageCurveDecoder(nn.Module):
    def __init__(self, manifold_dimension, reconstruct_dimension, hidden_dim=350):
        super(VariationalImageCurveDecoder, self).__init__()
        self.reconstruct_dimension = reconstruct_dimension
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        # self.rnn = nn.LSTM(manifold_dimension, hidden_dim, batch_first=True)
        # self.init_hidden = torch.nn.Parameter(torch.normal(0, 0.1, size=(1,hidden_dim)), requires_grad=True)
        # self.init_cell = torch.nn.Parameter(torch.normal(0, 0.1, size=(1,hidden_dim)), requires_grad=True) 
        # self.linear = nn.Linear(hidden_dim, reconstruct_dimension)
        # self.linear1 = nn.Linear(manifold_dimension, hidden_dim)
        # self.linear2 = nn.Linear(hidden_dim, reconstruct_dimension)
        self.linear_layers = torch.nn.Sequential(*[
            nn.Linear(manifold_dimension, hidden_dim, bias = True ),
            self.sigmoid,
            nn.Linear(hidden_dim, reconstruct_dimension, bias = False ),
        ])
    def forward(self, sample_curve_points):
        # batch_size = sample_curve_points.shape[0]
        # h_0 = self.init_hidden.repeat(1,batch_size,1)
        # c_0 = self.init_cell.repeat(1,batch_size,1)
        # rnn_out, (h_n, c_n) = self.rnn(sample_curve_points,  (h_0, c_0) )
        return torch.clamp(self.linear_layers(sample_curve_points), 0.0, 1.0)
class VariationalImageCurveEncoder(nn.Module):
    def __init__(self, output_dim = 250, bezier_order=3, sequence_length=5, input_channels=3, hidden_dim=500):
        super(VariationalImageCurveEncoder, self).__init__()
        self.output_dim = output_dim
        self.bezier_order = bezier_order
        self.sequence_length = sequence_length
        self.num_latent_vars = (bezier_order+1)*output_dim
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.conv2dlayers = torch.nn.Sequential(*[
          #  nn.BatchNorm2d(input_channels),
            nn.Conv2d(input_channels, 24, kernel_size=5, stride=2, bias=False),
            nn.BatchNorm2d(24),
            #self.relu,
            nn.Conv2d(24, 36, kernel_size=5, stride=2, bias=False),
            nn.BatchNorm2d(36),
            #self.relu,
            nn.Conv2d(36, 48, kernel_size=5, stride=2, bias=False),
            nn.BatchNorm2d(48), 
           # self.relu,
            nn.Conv2d(48, 64, kernel_size=3, bias=False),
            nn.BatchNorm2d(64),
            #self.relu,
            nn.Conv2d(64, 96, kernel_size=3),
           # self.tanh,
            nn.BatchNorm2d(96)
        ])
        self.rnn = nn.LSTM(1728, hidden_dim, batch_first=True)
        self.init_hidden = torch.nn.Parameter(torch.normal(0, 0.1, size=(1,hidden_dim)), requires_grad=True)
        self.init_cell = torch.nn.Parameter(torch.normal(0, 0.1, size=(1,hidden_dim)), requires_grad=True) 
        
        
        self.down_to_bezier_mu = nn.Linear(hidden_dim, self.num_latent_vars)
        self.down_to_bezier_logvar = nn.Linear(hidden_dim, self.num_latent_vars)
       
    def forward(self, images):
        batch_size = images.shape[0]
        assert(images.shape[1]==self.sequence_length)
        channels, rows, columns = images.shape[2:]
        packed_for_2dconv = images.view(-1,channels,rows,columns)

        conv2dout = self.conv2dlayers(packed_for_2dconv)
      #  print(conv2dout.shape)
        conv2doutflatten = conv2dout.view(batch_size, self.sequence_length, -1, )
       # print(conv2doutflatten.shape)
        h_0 = self.init_hidden.unsqueeze(1).repeat(1,batch_size,1)
        c_0 = self.init_cell.unsqueeze(1).repeat(1,batch_size,1)
        rnnout, (h_n, c_n) = self.rnn(conv2doutflatten, (h_0, c_0))
     #   print(rnnout.shape)


    
        #encoderout = self.encoder(images)
        bezier_mu_flat = self.down_to_bezier_mu(rnnout[:,-1])
        bezier_logstdev_flat = self.down_to_bezier_mu(rnnout[:,-1])
        bezier_stdev_flat = torch.exp(0.5*bezier_logstdev_flat)

        bezier_mu = bezier_mu_flat.view(batch_size, self.bezier_order+1, self.output_dim)
        bezier_stdev = bezier_stdev_flat.view(batch_size, self.bezier_order+1, self.output_dim)
        
        scale_tril = torch.diag_embed(bezier_stdev_flat)
       # distribution = torch.distributions.MultivariateNormal(, scale_tril=scale_tril, validate_args=True)
      # # curvesample = distribution.sample((1,))[0].view(batch_size, self.bezier_order+1, self.output_dim)

        return bezier_mu, bezier_stdev