from numpy import linalg
import torch.nn as nn 
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import torch
import torch.nn.utils.rnn as RNNUtils
import sys
import deepracing_models.math_utils as mu
import math



class ExternalAgentCurvePredictor(nn.Module):
    def __init__(self, output_dim : int = 2, bezier_order : int = 3, input_dim : int = 10, hidden_dim : int = 500, num_layers : int = 1, dropout : float = 0.0, bidirectional : bool = True, learnable_initial_state : bool = False):
        super(ExternalAgentCurvePredictor, self).__init__()
        self.output_dim : int = output_dim
        self.input_dim : int = input_dim
        self.hidden_dim : int = hidden_dim
        self.num_layers : int = num_layers
        self.bezier_order : int = bezier_order
        self.num_layers : int = num_layers
        self.dropout : float = dropout
        self.bidirectional : bool = bidirectional

        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, dropout=self.dropout, batch_first=True, bidirectional=self.bidirectional)
        self.fc = nn.Linear(self.hidden_dim*(int(self.bidirectional)+1), self.bezier_order*self.output_dim)
        if learnable_initial_state:
            self.init_hidden = Parameter(0.001*torch.randn((int(self.bidirectional)+1)*self.num_layers, self.hidden_dim))
            self.init_cell = Parameter(0.001*torch.randn((int(self.bidirectional)+1)*self.num_layers, self.hidden_dim))
        else:
            self.init_hidden = Parameter(torch.zeros((int(self.bidirectional)+1)*self.num_layers, self.hidden_dim), requires_grad=False)
            self.init_cell = Parameter(torch.zeros((int(self.bidirectional)+1)*self.num_layers, self.hidden_dim), requires_grad=False)


    def forward(self, x):
        batch_size = x.shape[0]
        init_hidden = self.init_hidden.unsqueeze(1).repeat(1,batch_size,1)
        init_cell = self.init_cell.unsqueeze(1).repeat(1,batch_size,1)
        lstm_out, (hidden, cell) = self.lstm(x, (init_hidden,init_cell))
        linear_out = self.fc(lstm_out[:,-1])
        return linear_out.view(batch_size, self.bezier_order, self.output_dim)

class ProbabilisticExternalAgentCurvePredictor(nn.Module):
    def __init__(self, output_dim : int = 2, bezier_order : int = 3, input_dim : int = 10, hidden_dim : int = 500, num_layers : int = 1, dropout : float = 0.0, bidirectional : bool = True, learnable_initial_state : bool = False):
        super(ProbabilisticExternalAgentCurvePredictor, self).__init__()
        self.output_dim : int = output_dim
        self.input_dim : int = input_dim
        self.hidden_dim : int = hidden_dim
        self.num_layers : int = num_layers
        self.bezier_order : int = bezier_order
        self.num_layers : int = num_layers
        self.dropout : float = dropout
        self.bidirectional : bool = bidirectional

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, dropout=self.dropout, batch_first=True, bidirectional=self.bidirectional)
        self.fc = nn.Linear(self.hidden_dim*(int(self.bidirectional)+1), self.bezier_order*self.output_dim)
        self.fc_variance = nn.Linear(100, self.bezier_order*self.output_dim)
        self.fc_variance.bias=Parameter(torch.ones(self.fc_variance.bias.shape[0]))
        self.fc_variance.weight=Parameter(0.001*torch.randn(self.fc_variance.weight.shape[0], self.fc_variance.weight.shape[1]))
        self.var_classifier = nn.Sequential(*[
            nn.Linear(self.hidden_dim*(int(self.bidirectional)+1), 250),
            self.relu,
            nn.Linear(250, 100),
            self.sigmoid,
            self.fc_variance,
            self.relu
            ]
        )

        self.num_covars = int((self.output_dim*(self.output_dim - 1))/2)
        self.fc_covariance = nn.Linear(100, self.bezier_order*self.num_covars)
        self.fc_covariance.bias=Parameter(0.00001*torch.randn(self.fc_covariance.bias.shape[0]))
        self.fc_covariance.weight=Parameter(0.00001*torch.randn(self.fc_covariance.weight.shape[0], self.fc_covariance.weight.shape[1]))
        self.covar_classifier = nn.Sequential(*[
            nn.Linear(self.hidden_dim*(int(self.bidirectional)+1), 250),
            self.relu,
            nn.Linear(250, 100),
            self.sigmoid,
            self.fc_covariance,
            self.relu
            ]
        )


        if learnable_initial_state:
            self.init_hidden = Parameter(0.001*torch.randn((int(self.bidirectional)+1)*self.num_layers, self.hidden_dim))
            self.init_cell = Parameter(0.001*torch.randn((int(self.bidirectional)+1)*self.num_layers, self.hidden_dim))
        else:
            self.init_hidden = Parameter(torch.zeros((int(self.bidirectional)+1)*self.num_layers, self.hidden_dim), requires_grad=False)
            self.init_cell = Parameter(torch.zeros((int(self.bidirectional)+1)*self.num_layers, self.hidden_dim), requires_grad=False)
    def forward(self, x):
        batch_size = x.shape[0]
        init_hidden = self.init_hidden.unsqueeze(1).repeat(1,batch_size,1)
        init_cell = self.init_cell.unsqueeze(1).repeat(1,batch_size,1)
        lstm_out, (hidden, cell) = self.lstm(x, (init_hidden,init_cell))
        means = self.fc(lstm_out[:,-1]).view(batch_size, self.bezier_order, self.output_dim)
        vfpred = self.var_classifier(lstm_out[:,-1]).view(batch_size, self.bezier_order, self.output_dim)
        first_point_variance = 1E-4*torch.ones_like(vfpred[:,0]).unsqueeze(1)


        cvfpred = self.covar_classifier(lstm_out[:,-1]).view(batch_size, self.bezier_order, self.num_covars)
        first_point_covariance = torch.zeros_like(cvfpred[:,0]).unsqueeze(1)


        return means, torch.cat([first_point_variance, vfpred], dim=1), torch.cat([first_point_covariance, cvfpred], dim=1)