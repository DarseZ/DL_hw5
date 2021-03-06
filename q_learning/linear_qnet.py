import numpy as np

import torch
import torch.nn as nn


class LinearQNet(nn.Module):
    def __init__(self, env, config):
        """
        A state-action (Q) network with a single fully connected
        layer, takes the state as input and gives action values
        for all actions.
        """
        super().__init__()

        #####################################################################
        # TODO: Define a fully connected layer for the forward pass. Some
        # useful information:
        #     observation shape: env.observation_space.shape -> (H, W, C)
        #     number of actions: env.action_space.n
        #     number of stacked observations in state: config.state_history
        #####################################################################
        H, W, C = env.observation_space.shape
        self.num_state_vals = H*W*C*config.state_history
        self.fullCnct = nn.Linear(self.num_state_vals, env.action_space.n)
        #####################################################################
        #                             END OF YOUR CODE                      #
        #####################################################################

    def forward(self, state):
        """
        Returns Q values for all actions

        Args:
            state: tensor of shape (batch, H, W, C x config.state_history)

        Returns:
            q_values: tensor of shape (batch_size, num_actions)
        """
        #####################################################################
        # TODO: Implement the forward pass, 1-2 lines.
        #####################################################################
        num_batch = state.size()[0]
        state = state.reshape(num_batch, -1)
        q_values = self.fullCnct(state)

        return q_values
        #####################################################################
        #                             END OF YOUR CODE                      #
        #####################################################################
