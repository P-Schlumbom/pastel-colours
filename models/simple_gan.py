import numpy as np
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, input_size, layer_size=64):
        super(Generator, self).__init__()

        self.lin1 = nn.Linear(input_size, layer_size)
        self.lin2 = nn.Linear(layer_size, layer_size)
        self.scaler = nn.Linear(layer_size, input_size)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x_init = x
        x = self.activation(self.lin1(x))
        x = self.activation(self.lin2(x))
        x = self.scaler(x) + x_init  # network should compute offset rather than new values, to encourage it to modify
        # input values rather than just memorising the correct answers. Needs linear scaling at the end as both input
        # and activation function output are in the range (-1, 1), and the modification needs to be a fraction of this.
        return x

class Discriminator(nn.Module):
    def __init__(self, input_size, layer_size=64):
        super(Discriminator, self).__init__()

        self.lin1 = nn.Linear(input_size, layer_size)
        self.lin2 = nn.Linear(layer_size, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.activation(self.lin1(x))
        x = self.activation(self.lin2(x))
        return x

