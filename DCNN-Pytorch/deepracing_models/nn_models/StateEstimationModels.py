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



class ExternalAgentCurvePredictor(nn.Module):
    def __init__(self, output_dim = 250, bezier_order=3, sequence_length=5, input_channels=3, hidden_dim=350):
        super(ExternalAgentCurvePredictor, self).__init__()