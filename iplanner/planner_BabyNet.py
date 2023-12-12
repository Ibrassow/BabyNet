
# ======================================================================
# Copyright (c) 2023 Fan Yang
# Robotic Systems Lab, ETH Zurich
# All rights reserved.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ======================================================================

import torch
from percept_net import ConvEncoder, Bottleneck, PerceptNet
import torch.nn as nn


class BabyNet(nn.Module):
    def __init__(self, encoder_channel=64, k=10):
        super().__init__()
        self.encoder = ConvEncoder(Bottleneck, layer_list=[2,6,8,4])
        self.decoder = DecoderLSTM(512, encoder_channel, k)

    def forward(self, x, goal):
        x = self.encoder(x)
        x, c = self.decoder(x, goal)
        return x, c



class DecoderLSTM(nn.Module):
    def __init__(self, in_channels, goal_channels, k=10, lstm_hidden_size=512):
        super().__init__()
        self.k = k
        self.relu    = nn.ReLU(inplace=True)
        self.gelu = nn.GELU()
        self.fg      = nn.Linear(3, goal_channels)
        self.sigmoid = nn.Sigmoid()

        # Convolutional layers
        self.conv1 = nn.Conv2d((in_channels + goal_channels), 512, kernel_size=5, stride=1, padding=1)
        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=0)

        # Fully connected layers
        self.fc1   = nn.Linear(256 * 128, 512) # to be used also for collision metric
        self.fc2   = nn.Linear(512, 512)
        self.fc3   = nn.Linear(512, lstm_hidden_size)

        self.dropout = torch.nn.Dropout(p=0.3)


        self.lstm_cells = torch.nn.Sequential(
          torch.nn.LSTMCell(input_size = lstm_hidden_size, hidden_size = lstm_hidden_size),
          torch.nn.LSTMCell(input_size = lstm_hidden_size, hidden_size = lstm_hidden_size),
          torch.nn.LSTMCell(input_size = lstm_hidden_size, hidden_size = lstm_hidden_size)
        )


        # Projection layer to coordinates
        self.projection = nn.Linear(lstm_hidden_size, 3)

        # Linear Layers for computation of the collision metric
        self.frc1 = nn.Linear(512, 256)
        self.frc2 = nn.Linear(256, 1)

        self.dropout_L = torch.nn.Dropout(p=0.2)



    def lstm_step(self, input_word, hidden_states_list):
      # Feed the input through each LSTM Cell
      for i in range(len(self.lstm_cells)):
          if i == 0:
              if hidden_states_list[i] is None:
                  hidden_states_list[i] = self.lstm_cells[i](input_word)
              else:
                  hidden_states_list[i] = self.lstm_cells[i](input_word, hidden_states_list[i])
          else:
              if hidden_states_list[i] is None:
                  hidden_states_list[i] = self.lstm_cells[i](hidden_states_list[i-1][0])
              else:
                  hidden_states_list[i] = self.lstm_cells[i](hidden_states_list[i-1][0], hidden_states_list[i])

          output = hidden_states_list[i][0]

          if self.training:
              output = self.dropout_L(output.unsqueeze(0)).squeeze(0)

          hidden_states_list[i] = (output, hidden_states_list[i][1])

      return hidden_states_list[-1][0], hidden_states_list



    def forward(self, x, goal):
        # Goal encoding
        goal = self.fg(goal[:, 0:3])
        goal = goal[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])

        # cat x with goal in channel dimension
        x = torch.cat((x, goal), dim=1)

        # Convolutional layers
        x = self.gelu(self.conv1(x))   # size = (N, 512, x.H/32, x.W/32)
        x = self.gelu(self.conv2(x))   # size = (N, 512, x.H/60, x.W/60)

        # Flatten and pass through fully connected layers
        x = torch.flatten(x, 1)
        f = self.gelu(self.dropout(self.fc1(x)))
        x = self.gelu(self.dropout(self.fc2(f)))
        x = self.gelu(self.dropout(self.fc3(x)))

        # Computes collision metric
        c = self.relu(self.frc1(f))
        c = self.sigmoid(self.frc2(c))

        hidden_states_list = [None]*len(self.lstm_cells)
        keypoints = [None]*self.k

        lstm_out = x

        for t in range(self.k): ## Number of keypoint path is our timestep

          # Reshape for LSTM
          #lstm_input = x.unsqueeze(1) # Add a sequence dimension
          lstm_out, hidden_states_list = self.lstm_step(lstm_out, hidden_states_list)

          # Project into coordinate space
          coord = self.projection(lstm_out)
          keypoints[t] = coord


        # Reshaping
        keypoints = torch.stack(keypoints, dim=1)


        return keypoints, c



