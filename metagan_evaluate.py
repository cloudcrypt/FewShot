import torch
import gflags
import sys
from torch.utils.data import DataLoader
from helpers import *
import numpy as np
from omni_loader import OmniLoader

if __name__ == '__main__':
    Flags = gflags.FLAGS
    gflags.DEFINE_string("model_path", "model/metagan", "path to store model")
    gflags.DEFINE_string("test_path", "omniglot/", 'path of testing folder')
    gflags.DEFINE_integer("way", 20, "how much way one-shot learning")
    gflags.DEFINE_string("times", 400, "number of samples to test accuracy")

    Flags(sys.argv)
    
    omniglot_loader = OmniLoader(
        dataset_path="omniglot", use_augmentation=False, batch_size=8)
    omniglot_loader.split_train_datasets()

    model = torch.load(Flags.model_path + "/best_metagan_model.pt")
    model.eval()
    evaluation_accuracy = omniglot_loader.one_shot_test(model, Flags.way, 40, False)

    print('Final Evaluation Accuracy = ' + str(evaluation_accuracy))
