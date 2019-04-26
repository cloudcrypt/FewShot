import torch
from siamese_model import *
import time
import numpy as np
import gflags
import sys
from collections import deque
import os
from helpers import *
from torchvision import transforms
from omniglot_loader import OmniglotTrain, OmniglotTest
from Generator import Generator
from torch.utils.data import DataLoader


def sample_noise(batch_size):
    return to_var(torch.rand(batch_size, 256))


def train_metagan(trainLoader, testLoader, Flags):

    loss_fn = torch.nn.BCELoss()

    net = SiameseNetwork(num_outputs=2)
    generator = Generator(image_size=Flags.image_size)

    if torch.cuda.is_available():
        print("Using CUDA")
        net.cuda()
        generator.cuda()

    D_optimizer = torch.optim.Adam(net.parameters(), lr=Flags.lr, betas=[Flags.beta1, Flags.beta2])
    G_optimizer = torch.optim.Adam(generator.parameters(), lr=Flags.lr, betas=[Flags.beta1, Flags.beta2])
    
    D_optimizer.zero_grad()
    G_optimizer.zero_grad()

    loss_val = 0
    time_start = time.time()
    support_set_size = 20

    best_validation_accuracy = 0.0
    best_accuracy_iteration = 0

    for iteration, (img1, img2, label) in enumerate(trainLoader, 1):
        if iteration > Flags.max_iter:
            break

        img1, img2, labels = to_var(img1), to_var(img2), to_var(label)

        net.train()
        generator.train()

        # images, labels = omniglot_loader.get_train_batch()
        # img1, img2, label = to_var(images[0]), to_var(images[1]), to_var(labels)

        D_optimizer.zero_grad()

        output = net.forward(img1, img2)

        D_real_pair_loss = loss_fn(output[:, 0].view(-1, 1), labels)
        D_real_disrcim_loss = loss_fn(output[:, 1].view(-1, 1), to_var(torch.ones((len(labels), 1))))

        noise = sample_noise(len(labels))
        img1_fake = generator.forward(img1, noise)
        img2_fake = generator.forward(img2, noise)
        fake_output = net.forward(img1_fake, img2_fake)
        del noise, img1_fake, img2_fake

        D_fake_discrim_loss = loss_fn(fake_output[:, 1].view(-1, 1), to_var(torch.zeros((len(labels), 1))))
        del fake_output

        D_total_loss = D_real_pair_loss + D_real_disrcim_loss + D_fake_discrim_loss
        loss_val += D_total_loss.item()
        D_total_loss.backward()
        D_optimizer.step()
        del D_total_loss, D_real_pair_loss, D_real_disrcim_loss, D_fake_discrim_loss

        G_optimizer.zero_grad()

        noise = sample_noise(len(labels))
        img1_fake = generator.forward(img1, noise)
        img2_fake = generator.forward(img2, noise)
        fake_output = net.forward(img1_fake, img2_fake)
        del noise, img1_fake, img2_fake

        G_loss = loss_fn(fake_output[:, 1].view(-1, 1), to_var(torch.ones((len(labels), 1))))
        G_loss.backward()
        G_optimizer.step()
        del G_loss, fake_output

        noise = sample_noise(len(labels))
        img1_fake = generator.forward(img1, noise)
        img2_fake = generator.forward(img2, noise)
        fake_output = net.forward(img1_fake, img2_fake)
        del noise, img1_fake, img2_fake

        D_fake_pair_loss = loss_fn(fake_output[:, 0].view(-1, 1), labels)
        D_fake_pair_loss.backward()
        D_optimizer.step()
        del D_fake_pair_loss, fake_output

        if iteration % Flags.show_every == 0:
            print('[%d]\tloss:\t%.5f\ttime lapsed:\t%.2f s' % (
            iteration, loss_val / Flags.show_every, time.time() - time_start))
            loss_val = 0
            time_start = time.time()
        if iteration % Flags.test_every == 0:
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
                torch.save(net, Flags.model_path + "/best_metagan_old_model.pt")

                best_validation_accuracy = validation_accuracy
                best_accuracy_iteration = iteration

            print('*' * 70)
            print('[%d]\tTest set\tAccuracy:\t%f' % (iteration, validation_accuracy))
            print('*' * 70)

        # If accuracy does not improve for 10000 batches stop the training
        if iteration - best_accuracy_iteration > 10000:
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
    gflags.DEFINE_integer("image_size", 105, "image size (width)")
    gflags.DEFINE_integer("way", 20, "how much way one-shot learning")
    gflags.DEFINE_string("times", 400, "number of samples to test accuracy")
    gflags.DEFINE_integer("workers", 4, "number of dataLoader workers")
    gflags.DEFINE_integer("batch_size", 128, "number of batch size")
    gflags.DEFINE_float("lr", 0.001, "learning rate")
    gflags.DEFINE_float("beta1", 0.5, "beta 1")
    gflags.DEFINE_float("beta2", 0.9, "beta 2")
    gflags.DEFINE_integer("show_every", 100, "show result after each show_every iter.")
    gflags.DEFINE_integer("save_every", 1000, "save model after each save_every iter.")
    gflags.DEFINE_integer("test_every", 1000, "test model after each test_every iter.")
    gflags.DEFINE_integer("max_iter", 1000000, "number of iterations before stopping")
    gflags.DEFINE_string("model_path", "model/metagan", "path to store model")

    Flags(sys.argv)

    data_transforms = transforms.Compose([
        transforms.RandomAffine(15),
        transforms.ToTensor()
    ])

    trainSet = OmniglotTrain(Flags.train_path, transform=data_transforms)
    testSet = OmniglotTest(Flags.test_path, transform=transforms.ToTensor(), times=Flags.times, way=Flags.way)
    testLoader = DataLoader(testSet, batch_size=Flags.way, shuffle=False)

    trainLoader = DataLoader(trainSet, batch_size=Flags.batch_size, shuffle=False)

    acc = train_metagan(trainLoader, testLoader, Flags)
    print("Best validation accuracy:" + str(acc))