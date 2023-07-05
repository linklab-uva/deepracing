import torch.nn as nn 
import torch.nn.functional as F
import torch
import collections

### 
# MixNet model is a modified version of work published by Phillip Karle & Technical University of Munich on August 22, 2022 under GNU General Public License V3.
# Modified on various dates starting April 14, 2023
###  
class BezierMixNet(nn.Module):
    """Neural Network to predict the future trajectory of a vehicle based on its history.
    It predicts mixing weights for mixing the boundaries, the centerline and the raceline.
    Also, it predicts section wise constant accelerations and an initial velocity to
    compute the velocity profile from.
    """

    def __init__(self, params : dict):
        """Initializes a BezierMixNet object trained to predict Bezier curves."""
        super(BezierMixNet, self).__init__()


        
        velocity_input : bool = params["velocity_input"]
        acceleration_input : bool = params["acceleration_input"]
        
        # Input embedding layer:
        input_dimension = 2
        if velocity_input:
            input_dimension+=2
        if acceleration_input:
            input_dimension+=2
        self._ip_emb = torch.nn.Linear(input_dimension, params["encoder"]["in_size"])
        constrain_initial_velocity : bool = params["constrain_initial_velocity"]
        constrain_initial_acceleration : bool = params["constrain_initial_acceleration"]
        if constrain_initial_acceleration and (not constrain_initial_velocity):
            raise ValueError("Constraining acceleration but not velocity is currently unsupported")
        bezier_order : int = params["bezier_order"]
        ambient_space_dimension : int = params["ambient_space_dimension"]

        self.constrain_initial_velocity : bool = constrain_initial_velocity
        self.constrain_initial_acceleration : bool = constrain_initial_acceleration
        self.ambient_space_dimension : int = ambient_space_dimension
        self.bezier_order : int = bezier_order

        self._boundary_emb = torch.nn.Linear(ambient_space_dimension, params["encoder"]["in_size"])

        encoderdict : dict = params["encoder"]
        # History encoder LSTM:
        self._enc_hist = torch.nn.LSTM(
            encoderdict["in_size"],
            encoderdict["hidden_size"],
            encoderdict.get("num_layers", 1),
            batch_first=True,
        )

        # Boundary encoders:
        self._enc_left_bound = torch.nn.LSTM(
            params["encoder"]["in_size"],
            params["encoder"]["hidden_size"],
            encoderdict.get("num_layers", 1),
            batch_first=True,
        )

        self._enc_right_bound = torch.nn.LSTM(
            params["encoder"]["in_size"],
            params["encoder"]["hidden_size"],
            encoderdict.get("num_layers", 1),
            batch_first=True,
        )

        #There are order+1 points in total, but the first point is always the vehicle's initial position, which is known.
        num_output_control_points : int = bezier_order
        if constrain_initial_velocity:
            num_output_control_points-=1
        if constrain_initial_acceleration:
            num_output_control_points-=1
        
        self.num_output_control_points = num_output_control_points
        # Linear stack that outputs the unconstrained control points
        mixerdict : dict = params["control_points_linear_stack"]
        self._mix_out_layers = self._get_linear_stack(
            in_size=encoderdict["hidden_size"] * 3,
            hidden_sizes=mixerdict["hidden_sizes"],
            out_size=num_output_control_points*ambient_space_dimension,
            name="control_point_producer",
        )

        prediction_time : float = params["prediction_time"]
        self.prediction_time : float = prediction_time
        
        history_time : float = params["history_time"]
        self.history_time : float = history_time
        
        self._params : dict = params.copy()

    def forward(self, history : torch.Tensor, left_bound : torch.Tensor, right_bound : torch.Tensor):
        """Implements the forward pass of the model.
        args:
            hist: [tensor with shape=(batch_size, hist_len, 2)]
            left_bound: [tensor with shape=(batch_size, boundary_len, 2)]
            right_bound: [tensor with shape=(batch_size, boundary_len, 2)]
        returns:
            control_points: [tensor with shape=(batch_size, k, d)]: The predicted control points
            k is how many points we need to predict given the initial value constraints, d is the dimension of the ambient euclidean space.
        """
        batch = history.shape[0]
        position_history : torch.Tensor = history[:,:,0:2]
        velocity_history : torch.Tensor = history[:,:,2:]

        
        p0 : torch.Tensor = position_history[:,-1]
        initial_velocity : torch.Tensor = velocity_history[:,-1]
        p1 : torch.Tensor = p0 + self.prediction_time*initial_velocity/self.bezier_order

        p0_unsqueeze : torch.Tensor = p0.unsqueeze(-2)
        p1_unsqueeze : torch.Tensor = p1.unsqueeze(-2)

        # encoders:
        _, (hist_h, _) = self._enc_hist(self._ip_emb(history))
        _, (left_h, _) = self._enc_left_bound(self._boundary_emb(left_bound))
        _, (right_h, _) = self._enc_right_bound(
            self._boundary_emb(right_bound)
        )
        enc = torch.squeeze(torch.cat((hist_h, left_h, right_h), 2), dim=0)

        control_points_flat = self._mix_out_layers(enc)
        control_points_deltas = control_points_flat.view(batch, self.num_output_control_points, self.ambient_space_dimension)
        control_points_predicted = p1_unsqueeze.expand_as(control_points_deltas) + control_points_deltas

        #modify the rest of this machinery later
        # path mixture through softmax:
        # mix_out = torch.softmax(self._mix_out_layers(enc), dim=1)

        # initial velocity:
        # vel_out = self._vel_out_layers(torch.squeeze(hist_h, dim=0))
        # vel_out = torch.sigmoid(vel_out)
        # vel_out = vel_out * self._params["init_vel_linear_stack"]["max_vel"]

        # acceleration decoding:
        # dec_input = torch.relu(self._dyn_embedder(enc)).unsqueeze(dim=1)
        # dec_input = dec_input.repeat(
        #     1, self._params["acc_decoder"]["num_acc_sections"], 1
        # )
        # acc_out, _ = self._acc_decoder(dec_input)
        # acc_out = torch.squeeze(self._acc_out_layer(torch.relu(acc_out)), dim=2)
        # acc_out = torch.tanh(acc_out) * self._params["acc_decoder"]["max_acc"]
        return torch.cat([p0_unsqueeze, p1_unsqueeze, control_points_predicted], dim=1)

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


