import torch
import torch.nn as nn


class FCDQN(nn.Module):
    def __init__(self, input_shape, n_actions, n_hidden):
        """Instantiate a fully connected network

        Args:
            input_shape (tuple): Shape of the observation space
            n_actions (int): Number of actions
            n_hidden (list): List of number of hidden units in each hidden layer
        """

        super(FCDQN, self).__init__()

        layers = [nn.Linear(input_shape[0], n_hidden[0]), nn.ReLU()]

        if len(n_hidden) > 1:
            for i in range(1, len(n_hidden)):
                layers.append(nn.Linear(n_hidden[i - 1], n_hidden[i]), nn.ReLU())
        layers.append(nn.Linear(n_hidden[-1], n_actions))

        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)