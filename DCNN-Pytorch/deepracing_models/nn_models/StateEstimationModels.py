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
    def __init__(self, output_dim : int = 3, bezier_order : int = 3, input_dim : int = 10, hidden_dim : int = 500, num_layers : int = 1, dropout : float = 0.0, bidirectional : bool = True):
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
        # self.init_hidden = Parameter(0.01*torch.randn((int(self.bidirectional)+1)*self.num_layers, self.hidden_dim))
        # self.init_cell = Parameter(0.01*torch.randn((int(self.bidirectional)+1)*self.num_layers, self.hidden_dim))

    def forward(self, x):
        batch_size = x.shape[0]
        lstm_out, (hidden, cell) = self.lstm(x)#, (init_hidden,init_cell))
        linear_out = self.fc(lstm_out[:,-1])
        return linear_out.view(batch_size, self.bezier_order, self.output_dim)