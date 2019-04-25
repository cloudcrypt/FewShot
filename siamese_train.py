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
from omni_loader import OmniLoader

if __name__ == '__main__':

    Flags = gflags.FLAGS
    gflags.DEFINE_string("train_path", "omniglot/", "training folder")
    gflags.DEFINE_string("test_path", "omniglot/", 'path of testing folder')
    gflags.DEFINE_integer("way", 20, "how much way one-shot learning")
    gflags.DEFINE_string("times", 400, "number of samples to test accuracy")
    gflags.DEFINE_integer("workers", 4, "number of dataLoader workers")
    gflags.DEFINE_integer("batch_size", 128, "number of batch size")
    gflags.DEFINE_float("lr", 0.00006, "learning rate")
    gflags.DEFINE_integer("show_every", 100, "show result after each show_every iter.")
    gflags.DEFINE_integer("save_every", 1000, "save model after each save_every iter.")
    gflags.DEFINE_integer("test_every", 1000, "test model after each test_every iter.")
    gflags.DEFINE_integer("max_iter", 1000000, "number of iterations before stopping")
    gflags.DEFINE_string("model_path", "model/siamese", "path to store model")

    Flags(sys.argv)

    if not os.path.exists(Flags.model_path):
        os.makedirs(Flags.model_path)

    omniglot_loader = OmniLoader(
        dataset_path="omniglot", use_augmentation=True, batch_size=32)
    omniglot_loader.split_train_datasets()

    loss_fn = torch.nn.BCEWithLogitsLoss(size_average=True)
    net = SiameseNetwork()

    if torch.cuda.is_available():
        net.cuda()

    net.train()

    optimizer = torch.optim.Adam(net.parameters(),lr = Flags.lr )
    optimizer.zero_grad()

    train_loss = []
    loss_val = 0
    time_start = time.time()
    queue = deque(maxlen=20)
    support_set_size = 20
    
    best_accuracy = 0.0

    for iteration in range(Flags.max_iter):
        images, labels = omniglot_loader.get_train_batch()
        img1, img2, label = to_var(images[0]), to_var(images[1]), to_var(labels)

        optimizer.zero_grad()
        output = net.forward(img1, img2)
        loss = loss_fn(output, label)
        loss_val += loss.item()
        loss.backward()
        optimizer.step()
        if iteration % Flags.show_every == 0 :
            print('[%d]\tloss:\t%.5f\ttime lapsed:\t%.2f s'%(iteration, loss_val/Flags.show_every, time.time() - time_start))
            loss_val = 0
            time_start = time.time()
        if iteration % Flags.save_every == 0:
            torch.save(net.state_dict(), Flags.model_path + '/model-inter-' + str(iteration+1) + ".pt")
        if iteration % Flags.test_every == 0:
            global_accuracy = 0.0
            number_of_runs_per_alphabet = 40
            validation_accuracy = omniglot_loader.one_shot_test(
                net, support_set_size, number_of_runs_per_alphabet, is_validation=True)
            
            if validation_accuracy > best_accuracy:
                if not os.path.exists(Flags.model_path):
                    os.makedirs(Flags.model_path)
                torch.save(net.state_dict(), Flags.model_path + "/best_siamese_model.pt")
                
                best_accuracy = validation_accuracy
            queue.append(global_accuracy)
        train_loss.append(loss_val)

    with open('train_loss', 'wb') as f:
        pickle.dump(train_loss, f)

    print("Best validation accuracy:"+ str(best_accuracy))

    acc = 0.0
    for d in queue:
        acc += d
    print("#"*70)
    print("final accuracy: ", acc/20)
