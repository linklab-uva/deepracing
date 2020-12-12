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

class VariationalCurvePredictor(nn.Module):
    def __init__(self, input_channels=3, params_per_dimension=11, \
                 context_length = 5, hidden_dim = 200, num_recurrent_layers = 1, rnn_bidirectional=False,  \
                    additional_rnn_calls=25, learnable_initial_state=True, output_dimension = 2, use_3dconv=True):
        super(VariationalCurvePredictor, self).__init__()
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
        self.mean_classifier = nn.Sequential(*[
            nn.Linear(self.hidden_decoder_features, 1200),
            self.relu,
            nn.Linear(1200, 500),
            self.tanh,
            nn.Linear(500, self.params_per_dimension)
            #nn.Linear(2432, self.params_per_dimension)
            ]
        )
        numvars = self.output_dimension*self.params_per_dimension
        numcovars = (self.output_dimension-1)*self.params_per_dimension
        self.classifier = nn.Sequential(*[
            nn.Linear(self.hidden_decoder_features, 1200),
            self.relu,
            nn.Linear(1200, 500),
            self.tanh,
            nn.Linear(500, self.params_per_dimension)
            #nn.Linear(2432, self.params_per_dimension)
            ]
        )
        self.var_linear = nn.Linear(250, self.params_per_dimension)
        self.var_linear.bias=torch.nn.Parameter(0.5*torch.ones(self.params_per_dimension), requires_grad=True)
        self.var_classifier = nn.Sequential(*[
            nn.Linear(self.hidden_decoder_features, 1200),
            self.relu,
            nn.Linear(1200, 250),
            self.tanh,
            self.var_linear,
            self.relu
            ]
        )
        self.covar_classifier = nn.Sequential(*[
            nn.Linear(self.hidden_decoder_features, 1200),
            self.relu,
            nn.Linear(1200, 250),
            self.tanh,
            nn.Flatten(),
            nn.Linear(self.output_dimension*250, numcovars)
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

        means = self.classifier(x_features).transpose(1,2)
        varfactors = (self.var_classifier(x_features) + 1E-3).transpose(1,2)
        covarfactors = self.covar_classifier(x_features)

        return means, varfactors, covarfactors

class ConvolutionalAutoencoder(nn.Module):
    def __init__(self, manifold_channels, in_channels):
        super(ConvolutionalAutoencoder, self).__init__()
        self.manifold_channels = manifold_channels
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
            nn.Conv2d(64, manifold_channels, kernel_size = 12, bias = True),
            self.elu,
        ])
        
        self.deconv_layers = nn.Sequential(*[
            nn.ConvTranspose2d(manifold_channels, 64, kernel_size = 12, bias = True),
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
    #    print(z.shape)
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