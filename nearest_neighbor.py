import torch
import torch.nn as nn

class NearestNeighbor(nn.Module) :
    def __init__(self):
        super(NearestNeighbor,self).__init__()

    def forward(self, image1, image2):
        batch_size = 1
        im_width = 0
        im_height = 0
        if len(image1.shape) == 4:
            batch_size = image1.shape[0]
            im_width = image1.shape[2]
            im_height = image1.shape[3]
        else:
            im_width = image1.shape[1]
            im_height = image1.shape[2]

        im1 = image1.reshape(batch_size, im_width*im_height)
        im2 = image2.reshape(batch_size, im_width*im_height)

        return torch.reciprocal(torch.norm(im1-im2, dim=1))
