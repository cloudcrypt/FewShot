import torch
import torch.nn as nn
from InstanceEncoder import InstanceEncoder

class Generator(nn.Module):
    def __init__(self, image_size=105):
        super(Generator, self).__init__()

        self.image_size = image_size

        self.instance_encoder = InstanceEncoder(image_size=self.image_size)

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=4),
            nn.Conv2d(512, 512, 3),
            nn.LeakyReLU(), # 512@2x2
            nn.Upsample(scale_factor=4),
            nn.Conv2d(512, 256, 3),
            nn.LeakyReLU(), #256@6x6
            nn.Upsample(scale_factor=4),
            nn.Conv2d(256, 128, 3),
            nn.LeakyReLU(), #128@22x22
            nn.Upsample(scale_factor=4),
            nn.Conv2d(128, 128, 3),
            nn.LeakyReLU(), #128@86x86
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 1, 3),
            nn.LeakyReLU(), #1@170x170
        )

        self.resize = nn.functional.interpolate

    
    def forward(self, x, z):
        """
        x: BS * 1 * 105 * 105
        z: BS * 256
        """
        x = self.instance_encoder(x)
        x = x.sum(dim=0) / x.shape[0]

        x = x.repeat(z.shape[0], 1)
        x = torch.cat((x,z), dim=1)
        x = x.view(x.shape[0], -1, 1, 1)

        x = self.up(x)

        x = self.resize(x, size=(self.image_size, self.image_size))

        return x
