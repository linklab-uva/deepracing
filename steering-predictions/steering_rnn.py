# Copyright (c) 2016-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

## @package char_rnn
# Module caffe2.python.examples.char_rnn
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.python.net_builder import ops
from caffe2.python import core, workspace, model_helper, utils, brew
from caffe2.python.rnn_cell import LSTM
from caffe2.proto import caffe2_pb2
from caffe2.python.optimizer import build_sgd
#from caffe2.python.predictor.mobile_exporter import Export, add_tensor
from caffe2.python.predictor.predictor_exporter import get_predictor_exporter_helper, PredictorExportMeta

import argparse
import logging
import numpy as np
from datetime import datetime




logging.basicConfig()
log = logging.getLogger("steering_rnn")
log.setLevel(logging.ERROR)


# Default set() here is intentional as it would accumulate values like a global
# variable
def CreateNetOnce(net, created_names=set()): # noqa
    name = net.Name()
    if name not in created_names:
        created_names.add(name)
        workspace.CreateNet(net)


class SteeringRNN(object):
    def __init__(self, args):
        self.seq_length = args.seq_length
        self.batch_size = 1
        self.iters_to_report = args.iters_to_report
        self.hidden_size = args.hidden_size

        with open(args.train_data) as f:
            self.text = f.read()

        self.vocab = list(set(self.text))
        self.char_to_idx = {ch: idx for idx, ch in enumerate(self.vocab)}
        self.idx_to_char = {idx: ch for idx, ch in enumerate(self.vocab)}
        self.D = len(self.char_to_idx)
        self.deepfeature_length = 500
	self.num_variables=10
	self.input_channels = 3
	self.input_width = 256
	self.input_height = 256

        print("Input has {} characters. Total input size: {}".format(
            len(self.vocab), len(self.text)))

    def CreateModel(self):
        log.debug("Start training")
        model = model_helper.ModelHelper(name="steering_rnn")

        input_blob, seq_lengths, hidden_init, cell_init, target = \
            model.net.AddExternalInputs(
                'input_blob',
                'seq_lengths',
                'hidden_init',
                'cell_init',
                'target',
            )
	''''''
	    # Image size: 256 X 256 -> 248 X 248
	self.conv0 = brew.conv(model, input_blob, 'conv0', dim_in=self.input_channels, dim_out=20, kernel=5)
	    # Image size: 248 X 248 -> 124 x 124
	self.pool0 = brew.max_pool(model, self.conv0, 'pool0', kernel=2, stride=2)
	    # Image size: 124 x 124 -> 120 x 120
	self.conv1 = brew.conv(model, self.pool0, 'conv1', dim_in=20, dim_out=40, kernel=5)
	    # Image size: 120 x 120 -> 60 x 60
	self.pool1 = brew.max_pool(model, self.conv1, 'pool1', kernel=2, stride=2)
	    # Image size: 60 x 60 -> 56 x 56
	self.conv2 = brew.conv(model, self.pool1, 'conv2', dim_in=40, dim_out=60, kernel=5)
	    # Image size: 56 x 56 -> 28 x 28
	self.pool2 = brew.max_pool(model, self.conv2, 'pool2', kernel=2, stride=2)
	    # Image size: 28 x 28 -> 24 x 24
	self.conv3 = brew.conv(model, self.pool2, 'conv3', dim_in=60, dim_out=80, kernel=5)
	    # Image size: 24 x 24 -> 12 x 12
	self.pool3 = brew.max_pool(model, self.conv3, 'pool3', kernel=2, stride=2)
	    # Image size: 12 x 12 -> 8 x 8
	self.conv4 = brew.conv(model, self.pool3, 'conv4', dim_in=80, dim_out=100, kernel=5)
	    # Image size: 8 x 8 -> 4 x 4
	self.pool4 = brew.max_pool(model, self.conv4, 'pool4', kernel=2, stride=2)
	    # Flatten from 100 * 4 * 4 image length to the "deep feature" vector
	self.fc_conv = brew.fc(model, self.pool4, 'fc_conv', dim_in=100 * 4 * 4, dim_out=self.deepfeature_length)	
	    # Reshape to fit what the LSTM implementation in caffe2 expects.
	fc_conv_reshaped, _ = model.net.Reshape('fc_conv', ['fc_conv_reshaped', '_'], shape=[self.seq_length, 1, self.deepfeature_length])
	self.fc_conv_reshaped=fc_conv_reshaped
        hidden_output_all, self.hidden_output, _, self.cell_state = LSTM(
            model, self.fc_conv_reshaped, seq_lengths, (hidden_init, cell_init),
            self.deepfeature_length, self.hidden_size, scope="LSTM")
        fc_recurrent = brew.fc(
            model,
            hidden_output_all,
            'fc_recurrent',
            dim_in=self.hidden_size,
            dim_out=self.num_variables,
            axis=2, 
	    debug_info=False
        )
        self.predictions = fc_recurrent
        # axis is 2 as first two are T (time) and N (batch size).
        # We treat them as one big batch of size T * N
        fc_recurrent_reshaped, _ = model.net.Reshape(
            'fc_recurrent', ['fc_recurrent_reshaped', '__'], shape=[-1, self.num_variables])
        self.output_reshaped = fc_recurrent_reshaped
        target_reshaped, _ = model.net.Reshape(
            'target', ['target_reshaped', '___'], shape=[-1, self.num_variables])
        self.target_reshaped = target_reshaped

        # Create a copy of the current net. We will use it on the forward
        # pass where we don't need loss and backward operators
        self.forward_net = core.Net(model.net.Proto())

        self.squared_norms = model.net.SquaredL2Distance([fc_recurrent_reshaped, target_reshaped], 'l2_norms')
        # Loss is average both across batch and through time
        # Thats why the learning rate below is multiplied by self.seq_length
        self.loss = model.net.AveragedLoss(self.squared_norms, 'loss')
        model.AddGradientOperators([self.loss])

        # use build_sdg function to build an optimizer
        build_sgd(
            model,
            base_learning_rate=0.1 * self.seq_length,
            policy="step",
            stepsize=1,
            gamma=0.9999
        )

        self.model = model

        self.prepare_state = core.Net("prepare_state")
        self.prepare_state.Copy(self.hidden_output, hidden_init)
        self.prepare_state.Copy(self.cell_state, cell_init)
 	#print(model.net.Proto())

    def _idx_at_pos(self, pos):
        return self.char_to_idx[self.text[pos]]

    def TrainModel(self):
        log.debug("Training model")

        workspace.RunNetOnce(self.model.param_init_net)

        # As though we predict the same probability for each character
        smooth_loss = -np.log(1.0 / self.D) * self.seq_length
        last_n_iter = 0
        last_n_loss = 0.0
        num_iter = 0
        N = len(self.text)
	

        # Writing to output states which will be copied to input
        # states within the loop below
        workspace.FeedBlob(self.hidden_output, np.zeros(
            [1, self.batch_size, self.hidden_size], dtype=np.float32
        ))
        workspace.FeedBlob(self.cell_state, np.zeros(
            [1, self.batch_size, self.hidden_size], dtype=np.float32
        ))
        workspace.CreateNet(self.prepare_state)

        # We iterate over text in a loop many times. Each time we peak
        # seq_length segment and feed it to LSTM as a sequence
        last_time = datetime.now()
        progress = 0
        while True:
            workspace.FeedBlob(
                "seq_lengths",
                np.array([self.seq_length] * self.batch_size,
                         dtype=np.int32)
            )
            workspace.RunNet(self.prepare_state.Name())

            input = np.random.rand(
                self.seq_length, self.input_channels, self.input_width, self.input_height
            ).astype(np.float32)
            target = np.random.rand(
                self.seq_length, self.batch_size, self.num_variables
            ).astype(np.float32)
	    '''
            for e in range(self.batch_size):
                for i in range(self.seq_length):
                    pos = text_block_starts[e] + text_block_positions[e]
                    input[i][e][self._idx_at_pos(pos)] = 1
                    target[i * self.batch_size + e] =\
                        self._idx_at_pos((pos + 1) % N)
                    text_block_positions[e] = (
                        text_block_positions[e] + 1) % text_block_sizes[e]
                    progress += 1
 	    '''

            workspace.FeedBlob('input_blob', input)
            workspace.FeedBlob('target', target)
           

            CreateNetOnce(self.model.net)
            workspace.RunNet(self.model.net.Name())

            predictions_out = workspace.FetchBlob(self.predictions)
            deep_features = workspace.FetchBlob(self.fc_conv)
            ''' '''
	    print("Input shape:", input.shape)
            print("Input:", input)

	    print("Deep Feature shape:", deep_features.shape)
            print("Deep Feature:", deep_features)

	    print("Predicted Output shape:", predictions_out.shape)
	    print("Predicted Output:", predictions_out)
	     
	    '''
	    init_pb, predictor_pb = Export(workspace, self.model.net, self.model.GetParams())
	    
	    print(predictor_pb)
	    print(init_pb)
	    '''
	    break          
	    
            num_iter += 1
            last_n_iter += 1

            if num_iter % self.iters_to_report == 0:
                new_time = datetime.now()
                print("Characters Per Second: {}". format(
                    int(progress / (new_time - last_time).total_seconds())
                ))
                print("Iterations Per Second: {}". format(
                    int(self.iters_to_report /
                        (new_time - last_time).total_seconds())
                ))

                last_time = new_time
                progress = 0

                print("{} Iteration {} {}".
                      format('-' * 10, num_iter, '-' * 10))

            loss = workspace.FetchBlob(self.loss) * self.seq_length
            smooth_loss = 0.999 * smooth_loss + 0.001 * loss
            last_n_loss += loss

            if num_iter % self.iters_to_report == 0:
                self.GenerateText(500, np.random.choice(self.vocab))

                log.debug("Loss since last report: {}"
                          .format(last_n_loss / last_n_iter))
                log.debug("Smooth loss: {}".format(smooth_loss))

                last_n_loss = 0.0
                last_n_iter = 0

    def GenerateText(self, num_characters, ch):
        # Given a starting symbol we feed a fake sequence of size 1 to
        # our RNN num_character times. After each time we use output
        # probabilities to pick a next character to feed to the network.
        # Same character becomes part of the output
        CreateNetOnce(self.forward_net)

        text = '' + ch
        for _i in range(num_characters):
            workspace.FeedBlob(
                "seq_lengths", np.array([1] * self.batch_size, dtype=np.int32))
            workspace.RunNet(self.prepare_state.Name())

            input = np.zeros([1, self.batch_size, self.D]).astype(np.float32)
            input[0][0][self.char_to_idx[ch]] = 1

            workspace.FeedBlob("input_blob", input)
            workspace.RunNet(self.forward_net.Name())

            p = workspace.FetchBlob(self.predictions)
            next = np.random.choice(self.D, p=p[0][0])

            ch = self.idx_to_char[next]
            text += ch

        print(text)


@utils.debug
def main():
    parser = argparse.ArgumentParser(
        description="Caffe2: Char RNN Training"
    )
    parser.add_argument("--train_data", type=str, default=None,
                        help="Path to training data in a text file format",
                        required=True)
    parser.add_argument("--seq_length", type=int, default=25,
                        help="One training example sequence length")
    parser.add_argument("--iters_to_report", type=int, default=500,
                        help="How often to report loss and generate text")
    parser.add_argument("--hidden_size", type=int, default=100,
                        help="Dimension of the hidden representation")
    parser.add_argument("--gpu", action="store_true",
                        help="If set, training is going to use GPU 0")

    args = parser.parse_args()

    device = core.DeviceOption(
        caffe2_pb2.CUDA if args.gpu else caffe2_pb2.CPU, 0)
    with core.DeviceScope(device):
        model = SteeringRNN(args)
        model.CreateModel()
        model.TrainModel()


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=2'])
    main()
