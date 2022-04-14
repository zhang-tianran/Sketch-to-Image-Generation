import torch.nn as nn
import torch.nn.functional as F
import torch

class Generator(nn.Module):
    def __init__(self, input_shape):
        super(Generator, self).__init__()

        # TODO
        layers = []
        layers.append()

        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        
        # TODO
        layers = []
        layers.append(nn.Linear())

        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)