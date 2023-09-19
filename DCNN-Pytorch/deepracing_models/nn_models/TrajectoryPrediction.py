import torch.nn as nn 
import torch.nn.functional as F
import torch
import collections

### 
# BezierMixNet is a modified version of MixNet, published by Phillip Karle & Technical University of Munich on August 22, 2022 under GNU General Public License V3.
# Modified on various dates starting April 14, 2023
###  
class BezierMixNet(nn.Module):

    def __init__(self, params : dict):
        """Initializes a BezierMixNet object."""
        super(BezierMixNet, self).__init__()

        self._params = params
        input_dimension = params["input_dimension"]


        self._inp_emb = torch.nn.Linear(input_dimension, params["encoder"]["in_size"])
        # History encoder LSTM:
        self._enc_hist = torch.nn.LSTM(
            params["encoder"]["in_size"],
            params["encoder"]["hidden_size"],
            1,
            batch_first=True,
        )

        # Boundary encoders:
        self._enc_left_bound = torch.nn.LSTM(
            params["encoder"]["in_size"],
            params["encoder"]["hidden_size"],
            1,
            batch_first=True,
        )

        self._enc_right_bound = torch.nn.LSTM(
            params["encoder"]["in_size"],
            params["encoder"]["hidden_size"],
            1,
            batch_first=True,
        )

        # Linear stack that outputs the path mixture ratios:
        self._mix_out_layers = self._get_linear_stack(
            in_size=params["encoder"]["hidden_size"] * 3,
            hidden_sizes=params["mixer_linear_stack"]["hidden_sizes"],
            out_size=params["mixer_linear_stack"]["out_size"],
            name="mix",
        )

        # dynamic embedder between the encoder and the decoder:
        self._dyn_embedder = nn.Linear(
            params["encoder"]["hidden_size"] * 3, params["acc_decoder"]["in_size"]
        )

        # acceleration decoder:
        self._acc_decoder = nn.LSTM(
            params["acc_decoder"]["in_size"],
            params["acc_decoder"]["hidden_size"],
            1,
            batch_first=True,
        )

        # output linear layer of the acceleration decoder:
        self._acc_out_layer = nn.Linear(params["acc_decoder"]["hidden_size"], 1)

        use_bias = True
        self._final_linear_layer = nn.Linear(4,4, bias=use_bias)
        self._final_linear_layer.weight = torch.nn.Parameter(torch.eye(4) + 0.0001*torch.randn(4,4))
        if use_bias:
            self._final_linear_layer.bias = torch.nn.Parameter(0.0001*torch.randn(4))
        # migrating the model parameters to the chosen device:
        if params["gpu_index"]>=0 and torch.cuda.is_available():
            self.device = torch.device("cuda:%d" % (params["gpu_index"],))
            print("Using CUDA as device for BezierMixNet")
        else:
            self.device = torch.device("cpu")
            print("Using CPU as device for BezierMixNet")
        
        self.to(self.device)

    def forward(self, hist, left_bound, right_bound):
        """Implements the forward pass of the model.

        args:
            hist: [tensor with shape=(batch_size, hist_len, hist_dim)]
            left_bound: [tensor with shape=(batch_size, boundary_len, 2)]
            right_bound: [tensor with shape=(batch_size, boundary_len, 2)]

        returns:
            mix_out: [tensor with shape=(batch_size, out_size)]: The path mixing ratios in the order:
                left_ratio, right_ratio, center_ratio, race_ratio
            vel_out: [tensor with shape=(batch_size, 1)]: The initial velocity of the velocity profile
            acc_out: [tensor with shape=(batch_size, num_acc_sections)]: The accelerations in the sections
        """

        # encoders:
        _, (hist_h, _) = self._enc_hist(self._inp_emb(hist.to(self.device)))
        _, (left_h, _) = self._enc_left_bound(self._inp_emb(left_bound.to(self.device)))
        _, (right_h, _) = self._enc_right_bound(self._inp_emb(right_bound.to(self.device)))

        # concatenate and squeeze encodings: 
        enc = torch.squeeze(torch.cat((hist_h, left_h, right_h), 2), dim=0)

        # path mixture through softmax:
        mix_out_softmax = torch.softmax(self._mix_out_layers(enc), dim=1)
        # mix_out_softmax = F.sigmoid(self._mix_out_layers(enc))
        # mix_out_softmax = self._mix_out_layers(enc)
        # mix_out = mix_out_softmax
        mix_out = self._final_linear_layer(mix_out_softmax)

        # acceleration decoding:
        dec_input = torch.relu(self._dyn_embedder(enc)).unsqueeze(dim=1)
        dec_input = dec_input.repeat(
            1, self._params["acc_decoder"]["num_acc_sections"] + 2, 1
        )
        acc_out, _ = self._acc_decoder(dec_input)
        acc_out = torch.squeeze(self._acc_out_layer(torch.relu(acc_out)), dim=2)
        acc_out = torch.tanh(acc_out) * self._params["acc_decoder"]["max_acc"]

        return mix_out, acc_out

    def load_model_weights(self, weights_path):
        self.load_state_dict(torch.load(weights_path, map_location=self.device))
        print("Successfully loaded model weights from {}".format(weights_path))

    def _get_linear_stack(
        self, in_size: int, hidden_sizes: list, out_size: int, name: str
    ):
        """Creates a stack of linear layers with the given sizes and with relu activation."""

        layer_sizes = []
        layer_sizes.append(in_size)  # The input size of the linear stack
        layer_sizes += hidden_sizes  # The hidden layer sizes specified in params
        layer_sizes.append(out_size)  # The output size specified in the params

        layer_list = []
        for i in range(len(layer_sizes) - 1):
            layer_name = name + "linear" + str(i + 1)
            act_name = name + "relu" + str(i + 1)
            layer_list.append(
                (layer_name, nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            )
            layer_list.append((act_name, nn.ReLU()))

        # removing the last ReLU layer:
        layer_list = layer_list[:-1]

        return nn.Sequential(collections.OrderedDict(layer_list))

    def get_params(self):
        """Accessor for the params of the network."""
        return self._params
