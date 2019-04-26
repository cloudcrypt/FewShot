import torch
from omni_loader import OmniLoader
import gflags
import sys

if __name__ == '__main__':
    Flags = gflags.FLAGS
    gflags.DEFINE_string("model_path", "model/siamese", "path to store model")

    Flags(sys.argv)

    omniglot_loader = OmniLoader(
        dataset_path="omniglot", use_augmentation=False, batch_size=32)
    omniglot_loader.split_train_datasets()

    model = torch.load(Flags.model_path + "/best_siamese_old_model.pt")
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    evaluation_accuracy = omniglot_loader.one_shot_test(model, 20, 40, False)

    print('Final Evaluation Accuracy = ' + str(evaluation_accuracy))
