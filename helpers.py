import torch
from torch.autograd import Variable

def to_torch_to_var(arr):
    return to_var(torch.from_numpy(arr).float())

def to_var(tensor, cuda=True):
    """Wraps a Tensor in a Variable, optionally placing it on the GPU.

        Arguments:
            tensor: A Tensor object.
            cuda: A boolean flag indicating whether to use the GPU.

        Returns:
            A Variable object, on the GPU if cuda==True.
    """
    if cuda and torch.cuda.is_available():
        return Variable(tensor.cuda())
    else:
        return Variable(tensor)

def to_data(x):
    """Converts variable to numpy."""
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data.numpy()