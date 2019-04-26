import torch
import torch.nn as nn


class SiameseNetwork(nn.Module) :
    def __init__(self, num_outputs=1):
        super(SiameseNetwork,self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 10),  # 64@96*96
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64@48*48
            nn.Conv2d(64, 128, 7),
            nn.ReLU(),    # 128@42*42
            nn.MaxPool2d(2),   # 128@21*21
            nn.Conv2d(128, 128, 4),
            nn.ReLU(), # 128@18*18
            nn.MaxPool2d(2), # 128@9*9
            nn.Conv2d(128, 256, 4),
            nn.ReLU(),   # 256@6*6
        )
        self.features = nn.Sequential(nn.Linear(9216, 4096), nn.Sigmoid())
        self.out = nn.Sequential(nn.Linear(4096, num_outputs), nn.Sigmoid())
        
    def forward_one(self, x):
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        x = self.features(x)
        return x

    def forward(self, image1, image2):
        out1 = self.forward_one(image1)
        out2 = self.forward_one(image2)
        dis = torch.abs(out1 - out2)
        out = self.out(dis)
        return out

