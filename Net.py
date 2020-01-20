import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import logging


class BaseNet(nn.Module):
    """Base class for all neural networks."""
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.rep_dim = None  # representation dimensionality, i.e. dim of the code layer or last layer

    def forward(self, *input):
        """
        Forward pass logic
        :return: Network output
        """
        raise NotImplementedError

    def summary(self):
        """Network summary."""
        net_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in net_parameters])
        self.logger.info('Trainable parameters: {}'.format(params))
        self.logger.info(self)


class MLP(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(MLP, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.out = torch.nn.Linear(n_hidden, n_output)       # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.out(x)
        return x


class MLPDrop(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(MLPDrop, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(n_feature, n_hidden),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(n_hidden, n_output)
        )

    def forward(self, x):
        x = self.classifier(x)
        return x


class MLP2Drop(torch.nn.Module):
    def __init__(self, n_feature, n_hidden1, n_hidden2, n_output):
        super(MLP2Drop, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(n_feature, n_hidden1),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(n_hidden1, n_hidden2),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(n_hidden2, n_output)
        )

    def forward(self, x):
        x = self.classifier(x)
        return x


class Net2(torch.nn.Module):
    def __init__(self, n_feature, n_hidden1, n_hidden2, n_output):
        super(Net2, self).__init__()
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden1)   # hidden layer
        self.hidden2 = torch.nn.Linear(n_hidden1, n_hidden2)  # hidden layer
        self.out = torch.nn.Linear(n_hidden2, n_output)       # output layer

    def forward(self, x):
        x = F.relu(self.hidden1(x))      # activation function for hidden layer
        x = F.relu(self.hidden2(x))      # activation function for hidden layer
        x = self.out(x)
        return x
