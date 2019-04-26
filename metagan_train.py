import torch
from siamese_model import *
import time
import numpy as np
import gflags
import sys
from collections import deque
import os
from helpers import *
from omni_loader import OmniLoader
from Generator import Generator

def sample_noise(batch_size):
    return to_var(torch.rand(batch_size, 256))

def train_metagan(omniglot_loader, Flags):

    loss_fn = torch.nn.BCELoss()

    net = SiameseNetwork(num_outputs=2)
    generator = Generator(image_size=Flags.image_size)

    if torch.cuda.is_available():
        print("Using CUDA")
        net.cuda()

    D_optimizer = torch.optim.Adam(net.parameters(), lr=Flags.lr, betas=[Flags.beta1, Flags.beta2])
    G_optimizer = torch.optim.Adam(generator.parameters(), lr=Flags.lr, betas=[Flags.beta1, Flags.beta2])
    
    D_optimizer.zero_grad()
    G_optimizer.zero_grad()

    loss_val = 0
    time_start = time.time()
    queue = deque(maxlen=20)
    support_set_size = 20

    best_validation_accuracy = 0.0
    best_accuracy_iteration = 0

    for iteration in range(Flags.max_iter):
        net.train()
        generator.train()

        images, labels = omniglot_loader.get_train_batch()
        img1, img2, label = to_var(images[0]), to_var(images[1]), to_var(labels)

        D_optimizer.zero_grad()

        output = net.forward(img1, img2)

        D_real_pair_loss = loss_fn(output[:, 0].view(-1, 1), label)
        D_real_disrcim_loss = loss_fn(output[:, 1].view(-1, 1), torch.ones((len(labels), 1)))

        noise = sample_noise(len(labels))
        img1_fake = generator.forward(img1, noise)
        img2_fake = generator.forward(img2, noise)
        fake_output = net.forward(img1_fake, img2_fake)

        D_fake_discrim_loss = loss_fn(fake_output[:, 1].view(-1, 1), torch.zeros((len(labels), 1)))

        D_total_loss = D_real_pair_loss + D_real_disrcim_loss + D_fake_discrim_loss
        loss_val += D_total_loss
        D_total_loss.backward()
        D_optimizer.step()

        G_optimizer.zero_grad()

        noise = sample_noise(len(labels))
        img1_fake = generator.forward(img1, noise)
        img2_fake = generator.forward(img2, noise)
        fake_output = net.forward(img1_fake, img2_fake)

        G_loss = loss_fn(fake_output[:, 1].view(-1, 1), torch.ones((len(labels), 1)))
        G_loss.backward()
        G_optimizer.step()

        noise = sample_noise(len(labels))
        img1_fake = generator.forward(img1, noise)
        img2_fake = generator.forward(img2, noise)
        fake_output = net.forward(img1_fake, img2_fake)

        D_fake_pair_loss = loss_fn(fake_output[:, 0].view(-1, 1), label)
        D_fake_pair_loss.backward()
        D_optimizer.step()

        if iteration % Flags.show_every == 0:
            print('[%d]\tloss:\t%.5f\ttime lapsed:\t%.2f s' % (
            iteration, loss_val / Flags.show_every, time.time() - time_start))
            loss_val = 0
            time_start = time.time()
        if iteration % Flags.test_every == 0:
            net.eval()

            global_accuracy = 0.0
            number_of_runs_per_alphabet = 40
            validation_accuracy = omniglot_loader.one_shot_test(
                net, support_set_size, number_of_runs_per_alphabet, is_validation=True)

            if validation_accuracy > best_validation_accuracy:
                if not os.path.exists(Flags.model_path):
                    os.makedirs(Flags.model_path)
                torch.save(net, Flags.model_path + "/best_metagan_model.pt")

                best_validation_accuracy = validation_accuracy
                best_accuracy_iteration = iteration
            queue.append(global_accuracy)

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

    omniglot_loader = OmniLoader(
        dataset_path="omniglot", use_augmentation=False, batch_size=32)
    omniglot_loader.split_train_datasets()

    acc = train_metagan(omniglot_loader, Flags)
    print("Best validation accuracy:" + str(acc))

    model = torch.load(Flags.model_path + "/best_metagan_model.pt")
    model.eval()
    evaluation_accuracy = omniglot_loader.one_shot_test(model, 20, 40, False)

    print('Final Evaluation Accuracy = ' + str(evaluation_accuracy))
