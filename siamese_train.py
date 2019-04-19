from helpers import *
from omniglot_loader import OmniglotLoader
import numpy as np
import torch
from torch import nn
from torch import optim
from siamese_model import SiameseNetwork
import os


def accuracy(output, target):
    """Computes the accuracy for multiple binary predictions"""
    pred = output >= 0.5
    truth = target >= 0.5
    acc = pred.eq(truth).float().mean()
    return np.asscalar(to_data(acc))

if __name__ == '__main__':

    omniglot_loader = OmniglotLoader(
        dataset_path="omniglot", use_augmentation=True, batch_size=32)

    omniglot_loader.split_train_datasets()

    evaluate_each = 1000
    train_losses = np.zeros(shape=evaluate_each)
    train_accuracies = np.zeros(shape=evaluate_each)
    count = 0
    earrly_stop = 0
    # Stop criteria variables
    best_validation_accuracy = 0.0
    best_accuracy_iteration = 0
    validation_accuracy = 0.0

    number_of_iterations = 1000000
    support_set_size = 20
    model_name = 'Siamese_Network'
    model = SiameseNetwork()
    if torch.cuda.is_available():
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=10e-4, weight_decay=0.0001)
    loss_fn = torch.nn.BCEWithLogitsLoss(size_average=True)

    model.train()
    for iteration in range(number_of_iterations):
        images, labels = omniglot_loader.get_train_batch()

        output = model(to_torch_to_var(images[0]), to_torch_to_var(images[1]))

        optimizer.zero_grad()

        train_loss = loss_fn(output, labels)
        train_loss.backward()
        optimizer.step()

        train_accuracy = accuracy(output, labels)

        # validation set
        if iteration % 100 == 0:
            print('Iteration %d/%d: Train loss: %f, Train Accuracy: %f' %
                 (iteration + 1, number_of_iterations, train_loss, train_accuracy))

        # Each 100 iterations perform a one_shot_task and write to tensorboard the
        # stored losses and accuracies
        if (iteration + 1) % evaluate_each == 0:
            number_of_runs_per_alphabet = 40
            # use a support set size equal to the number of character in the alphabet
            validation_accuracy = omniglot_loader.one_shot_test(
                model, support_set_size, number_of_runs_per_alphabet, is_validation=True)

            # Some hyperparameters lead to 100%, although the output is almost the same in
            # all images.
            if (validation_accuracy == 1.0 and train_accuracy == 0.5):
                print('Early Stopping: Gradient Explosion')
                print('Validation Accuracy = ' +
                      str(best_validation_accuracy))
                exit()
            elif train_accuracy == 0.0:
                exit()
            else:
                # Save the model
                if validation_accuracy > best_validation_accuracy:
                    best_validation_accuracy = validation_accuracy
                    best_accuracy_iteration = iteration

                    if not os.path.exists('./models'):
                        os.makedirs('./models')
                    torch.save(model.state_dict(), './models')

        # If accuracy does not improve for 10000 batches stop the training
        if iteration - best_accuracy_iteration > 10000:
            print(
                'Early Stopping: validation accuracy did not increase for 10000 iterations')
            print('Best Validation Accuracy = ' +
                  str(best_validation_accuracy))
            print('Validation Accuracy = ' + str(best_validation_accuracy))
            break
