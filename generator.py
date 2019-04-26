import torch
import torch.nn as nn
from InstanceEncoder import InstanceEncoder

class Generator(nn.Module):
    def __init__(self, image_size=105):
        super(Generator, self).__init__()

        self.instance_encoder = InstanceEncoder(image_size=image_size)

    def forward(self, x, z):
        x = self.instance_encoder(x)
        x = x.sum(dim=0) / x.shape[0]
        x = torch.cat((x,z))
        return x
