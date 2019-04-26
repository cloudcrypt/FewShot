import torch
import torch.nn as nn

class InstanceEncoder(nn.Module):
    def __init__(self, image_size=105):
        super(InstanceEncoder, self).__init__()

        self.image_size = image_size

        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 3),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 3),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, 3),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, 3, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 3),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, 3),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, 3, stride=2),
            nn.LeakyReLU()
        )

        convs = [(c.out_channels, c.kernel_size[0], c.stride[0]) for c in filter(lambda x: type(x) is nn.Conv2d, self.conv)]
        size = (1, self.image_size, self.image_size)

        for conv in convs:
            size = self.transform_size(size[1], *conv)

        linear_input_size = size[0] * size[1] * size[2]

        self.fc = nn.Sequential(
            nn.Linear(linear_input_size, 256),
            nn.LeakyReLU(),
            nn.Dropout()
        )

    def transform_size(self, width, conv_channels, conv_kernel_size, conv_stride):
        size = ((width - (conv_kernel_size - 1) - 1) // conv_stride) + 1
        return (conv_channels, size, size)

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size()[0], -1)
        out = self.fc(out)
        return out
