import torch
from omniglot_loader import OmniglotTrain, OmniglotTest
import gflags
import sys
from torch.utils.data import DataLoader
from helpers import *
import numpy as np


if __name__ == '__main__':
    Flags = gflags.FLAGS
    gflags.DEFINE_string("model_path", "model/siamese", "path to store model")
    gflags.DEFINE_string("test_path", "omniglot/", 'path of testing folder')
    gflags.DEFINE_integer("way", 20, "how much way one-shot learning")
    gflags.DEFINE_string("times", 400, "number of samples to test accuracy")

    Flags(sys.argv)

    testSet = OmniglotTest(Flags.test_path, times=Flags.times, way=Flags.way)
    testLoader = DataLoader(testSet, batch_size=Flags.way, shuffle=False)

    model = torch.load(Flags.model_path + "/best_siamese_old_model.pt")
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    evaluation_accuracy = 0.0
    for i in range(0, 800):
        img1, img2 = testSet.get_one_shot_batch()
        img1, img2 = to_var(img1), to_var(img2)
        output = model.forward(img1, img2)

        if np.asscalar(to_data(torch.argmax(output))) == 0:
            evaluation_accuracy += 1.0

    evaluation_accuracy /= 800.0

    print('Final Evaluation Accuracy = ' + str(evaluation_accuracy))
