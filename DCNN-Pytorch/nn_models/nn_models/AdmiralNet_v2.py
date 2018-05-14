class AdmiralNet_v2(nn.module,cell='lstm'):
    def __init__(self, sequence_length=25, context_length = 25, hidden_dim = 100, use_float32 = False, gpu = -1, optical_flow = False):
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
        
        self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2)
        
        #RESIDUAL BLOCK 2
        self.conv3 = nn.Conv2d(36, 48, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(48, 48, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(48, 48, kernel_size=3, stride=2)
        
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3)
        
        #RESIDUAL BLOCK 5
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3)
        self.conv5_2 = nn.Conv2d(64, 64, kernel_size=3)
        self.conv5_3 = nn.Conv2d(64, 64, kernel_size=3)
        
        #batch norm layers
        self.Norm_1 = nn.BatchNorm2d(24)
        self.Norm_2 = nn.BatchNorm2d(36)
        self.Norm_3 = nn.BatchNorm2d(48) 
        self.Norm_4 = nn.BatchNorm2d(64)
        
        #recurrent layers
        self.feature_length = 1*64*6*3
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.context_length = context_length
        
        #RNN support added
        if(cell=='lstm'):
            self.rnn = nn.LSTM(self.feature_length, hidden_dim, batch_first = True)
        elif(cell=='gru'):
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





