import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvQNet(nn.Module):
    def __init__(self, env, config, logger=None):
        super().__init__()

        #####################################################################
        # TODO: Define a CNN for the forward pass.
        #   Use the CNN architecture described in the following DeepMind
        #   paper by Mnih et. al.:
        #       https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
        #
        # Some useful information:
        #     observation shape: env.observation_space.shape -> (H, W, C)
        #     number of actions: env.action_space.n
        #     number of stacked observations in state: config.state_history
        #####################################################################
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels = 4, out_channels = 16, kernel_size = 8, stride = 4),
            nn.ReLU(),
            nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 4, stride = 2),
            nn.ReLU()
        )

        # fully connected layer to output Q-values for each action
        self.fc = nn.Linear(in_features = 2048, out_features = env.action_space.n)
        #####################################################################
        #                             END OF YOUR CODE                      #
        #####################################################################

    def forward(self, state):
        #####################################################################
        # TODO: Implement the forward pass.
        #####################################################################
        batch_size = state.shape[0]
        # transform the input into a CNN standard format
        state = state.permute(0, 3, 1, 2)
        # pass CNN sequential model
        conv_out = self.conv(state)
        # transform the input into a fc forward feed standard format
        conv_out = conv_out.reshape(batch_size, -1)
        # pass fc model
        fc_out = self.fc(conv_out)

        return fc_out
        #####################################################################
        #                             END OF YOUR CODE                      #
        #####################################################################
