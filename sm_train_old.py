import torch
import pickle
import torchvision
from torchvision import transforms
import torchvision.datasets as dset
from torchvision import transforms
from omniglot_loader import OmniglotTrain, OmniglotTest
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt
from siamese_model import *
import time
import numpy as np
import gflags
import sys
from collections import deque
import os
from helpers import *


def train_siamese(trainLoader, testLoader, Flags):
    loss_fn = torch.nn.BCEWithLogitsLoss(size_average=True)
    net = SiameseNetwork()

    if torch.cuda.is_available():
        net.cuda()

    net.train()

    optimizer = torch.optim.Adam(net.parameters(), lr=Flags.lr)
    optimizer.zero_grad()

    loss_val = 0
    time_start = time.time()

    best_validation_accuracy = 0.0
    best_accuracy_iteration = 0

    for batch_id, (img1, img2, label) in enumerate(trainLoader, 1):
        if batch_id > Flags.max_iter:
            break

        img1, img2, label = to_var(img1), to_var(img2), to_var(label)

        optimizer.zero_grad()
        output = net.forward(img1, img2)
        loss = loss_fn(output, label)
        loss_val += loss.item()
        loss.backward()
        optimizer.step()
        if batch_id % Flags.show_every == 0:
            print('[%d]\tloss:\t%.5f\ttime lapsed:\t%.2f s' % (
            batch_id, loss_val / Flags.show_every, time.time() - time_start))
            loss_val = 0
            time_start = time.time()
        if batch_id % Flags.test_every == 0:
            validation_accuracy = 0.0
            for i in range(0, 800):
                img1, img2 = testSet.get_one_shot_batch()
                img1, img2 = to_var(img1), to_var(img2)
                output = net.forward(img1, img2)

                if np.asscalar(to_data(torch.argmax(output))) == 0:
                    validation_accuracy += 1.0

            validation_accuracy /= 800.0
            if validation_accuracy > best_validation_accuracy:
                if not os.path.exists(Flags.model_path):
                    os.makedirs(Flags.model_path)
                torch.save(net, Flags.model_path + "/best_siamese_old_model.pt")

                best_validation_accuracy = validation_accuracy
                best_accuracy_iteration = batch_id

            print('*' * 70)
            print('[%d]\tTest set\tAccuracy:\t%f' % (batch_id, validation_accuracy))
            print('*' * 70)

        # If accuracy does not improve for 10000 batches stop the training
        if batch_id - best_accuracy_iteration > 10000:
            print(
                'Early Stopping: validation accuracy did not increase for 10000 iterations')
            print('Best Validation Accuracy = ' +
                  str(best_validation_accuracy))
            print('Validation Accuracy = ' + str(best_validation_accuracy))
            break

    return best_validation_accuracy


if __name__ == '__main__':

    Flags = gflags.FLAGS
    gflags.DEFINE_string("train_path", "omniglot/", "training folder")
    gflags.DEFINE_string("test_path", "omniglot/", 'path of testing folder')
    gflags.DEFINE_integer("way", 20, "how much way one-shot learning")
    gflags.DEFINE_string("times", 400, "number of samples to test accuracy")
    gflags.DEFINE_integer("workers", 4, "number of dataLoader workers")
    gflags.DEFINE_integer("batch_size", 128, "number of batch size")
    gflags.DEFINE_float("lr", 0.00006, "learning rate")
    gflags.DEFINE_integer("show_every", 10, "show result after each show_every iter.")
    gflags.DEFINE_integer("save_every", 100, "save model after each save_every iter.")
    gflags.DEFINE_integer("test_every", 100, "test model after each test_every iter.")
    gflags.DEFINE_integer("max_iter", 50000, "number of iterations before stopping")
    gflags.DEFINE_string("model_path", "model/siamese", "path to store model")

    Flags(sys.argv)

    data_transforms = transforms.Compose([
        transforms.RandomAffine(15),
        transforms.ToTensor()
    ])

    trainSet = OmniglotTrain(Flags.train_path, transform=data_transforms)
    testSet = OmniglotTest(Flags.test_path, transform=transforms.ToTensor(), times=Flags.times, way=Flags.way)
    testLoader = DataLoader(testSet, batch_size=Flags.way, shuffle=False, num_workers=Flags.workers)

    trainLoader = DataLoader(trainSet, batch_size=Flags.batch_size, shuffle=False, num_workers=Flags.workers)

    acc = train_siamese(trainLoader, testLoader, Flags)
    print("Best validation accuracy:" + str(acc))

