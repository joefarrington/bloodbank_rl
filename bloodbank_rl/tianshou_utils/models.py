import numpy as np
import torch
import torch.nn as nn

class FCDQN(nn.Module):
    def __init__(self, state_shape, action_shape, n_hidden, device="cpu"):
        """Instantiate a fully connected network

        Args:
            state_shape (tuple): Shape of the observation space
            action_shape (tuple): Shape of action space
            n_hidden (list): List of number of hidden units in each hidden layer
            device (str): defaults to "cpu", set to "cuda" if cuda is available
        """

        super(FCDQN, self).__init__()
        self.device = device

        layers = [nn.Linear(np.prod(state_shape), n_hidden[0]), nn.ReLU(inplace=True)]

        if len(n_hidden) > 1:
            for i in range(1, len(n_hidden)):
                layers.append(nn.Linear(n_hidden[i - 1], n_hidden[i]))
                layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(n_hidden[-1], np.prod(action_shape)))

        self.model = nn.Sequential(*layers)

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float, device=self.device)
        batch = obs.shape[0]
        logits = self.model(obs.view(batch, -1))
        return logits, state