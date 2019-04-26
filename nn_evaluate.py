import torch
from nearest_neighbor import *
from helpers import *
from omni_loader import *
if __name__ == '__main__':
    omniglot_loader = OmniLoader(
        dataset_path="omniglot", use_augmentation=False, batch_size=32)
    omniglot_loader.split_train_datasets()

    model = NearestNeighbor()
    model.eval()
    evaluation_accuracy = omniglot_loader.one_shot_test(model, 20, 40, False)

    print('Final Evaluation Accuracy = ' + str(evaluation_accuracy))
