import torch
import torch.nn as nn

class SiameseNetwork(nn.Module) :
    def __init__(self, batch_size):
        super(SiameseNetwork,self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, 10),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

    def forward(self, image1, image2):
        h1_1 = self.layer1(image1)

        return h1_1
