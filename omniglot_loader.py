import torch
from torch.utils.data import Dataset, DataLoader
import os
from numpy.random import choice as npc
import numpy as np
import time
import random
import torchvision.datasets as dset
from PIL import Image
import errno
from helpers import *
from torchvision import transforms

class OmniglotLoader:
    urls = [
        'https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip',
        'https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip'
    ]
    raw_folder = 'raw'
    processed_folder = 'processed'

    def __init__(self, path):
        self.dataset_path = path

    def download_if_necessary(self):
        if not self._check_exists():
            self.download()

    def _check_exists(self):
        return os.path.exists(os.path.join(self.dataset_path, self.processed_folder, "images_evaluation")) and \
               os.path.exists(os.path.join(self.dataset_path, self.processed_folder, "images_background"))

    def download(self):
        from six.moves import urllib
        import zipfile

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.dataset_path, self.raw_folder))
            os.makedirs(os.path.join(self.dataset_path, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            print('== Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.dataset_path, self.raw_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            file_processed = os.path.join(self.dataset_path, self.processed_folder)
            print("== Unzip from " + file_path + " to " + file_processed)
            zip_ref = zipfile.ZipFile(file_path, 'r')
            zip_ref.extractall(file_processed)
            zip_ref.close()
        print("Download finished.")


class OmniglotTrain(Dataset):

    def __init__(self, dataPath, batch_size=8, transform=None):
        super(OmniglotTrain, self).__init__()

        loader = OmniglotLoader(dataPath)
        loader.download_if_necessary()

        np.random.seed(0)
        # self.dataset = dataset
        self.transform = transform
        self.datas, self.num_classes = self.loadToMem(os.path.join(loader.dataset_path, loader.processed_folder, 'images_background'))
        self.batch_size = batch_size


    def loadToMem(self, dataPath):
        print("begin loading training dataset to memory")
        datas = {}
        agrees = [0, 90, 180, 270]
        idx = 0
        for agree in agrees:
            for alphaPath in os.listdir(dataPath):
                for charPath in os.listdir(os.path.join(dataPath, alphaPath)):
                    datas[idx] = []
                    for samplePath in os.listdir(os.path.join(dataPath, alphaPath, charPath)):
                        filePath = os.path.join(dataPath, alphaPath, charPath, samplePath)
                        datas[idx].append(Image.open(filePath).rotate(agree).convert('L'))
                    idx += 1
        print("finish loading training dataset to memory")
        return datas, idx

    def __len__(self):
        return  21000000

    def __getitem__(self, index):
        # image1 = random.choice(self.dataset.imgs)
        labels = torch.zeros(self.batch_size)
        img1 = torch.zeros(self.batch_size, 1, 105, 105)
        img2 = torch.zeros(self.batch_size, 1, 105, 105)
        for i in range(self.batch_size):
            if i % 2 == 1:
                idx1 = random.randint(0, self.num_classes - 1)
                idx2 = idx1
                labels[i] = 1.0
            else:
                idx1 = random.randint(0, self.num_classes - 1)
                idx2 = random.randint(0, self.num_classes - 1)
                labels[i] = 0.0
                while idx1 == idx2:
                    idx2 = random.randint(0, self.num_classes - 1)

            image1 = random.choice(self.datas[idx1])
            image2 = random.choice(self.datas[idx2])
            if self.transform:
                image1 = self.transform(image1)
                image2 = self.transform(image2)

            img1[i] = image1
            img2[i] = image2

        return img1, img2, labels


class OmniglotTest(Dataset):

    def __init__(self, dataPath, transform=None, times=200, way=20):
        np.random.seed(1)
        super(OmniglotTest, self).__init__()
        loader = OmniglotLoader(dataPath)
        loader.download_if_necessary()

        self.image_width = 105
        self.image_height = 105

        self.transform = transform
        self.times = times
        self.way = way
        self.img1 = None
        self.c1 = None
        self.datas, self.num_classes = self.loadToMem(os.path.join(loader.dataset_path, loader.processed_folder, 'images_evaluation'))

    def loadToMem(self, dataPath):
        print("begin loading test dataset to memory")
        datas = {}
        idx = 0
        for alphaPath in os.listdir(dataPath):
            for charPath in os.listdir(os.path.join(dataPath, alphaPath)):
                datas[idx] = []
                for samplePath in os.listdir(os.path.join(dataPath, alphaPath, charPath)):
                    filePath = os.path.join(dataPath, alphaPath, charPath, samplePath)
                    datas[idx].append(Image.open(filePath).convert('L'))
                idx += 1
        print("finish loading test dataset to memory")
        return datas, idx

    def get_one_shot_batch(self):
        if self.transform is None:
            transf = transforms.ToTensor()
        else:
            transf = self.transform

        images1 = torch.zeros(self.way+1, 1, self.image_width, self.image_height)

        test_class = random.randint(0, self.num_classes - 1)
        image_indices = random.sample(range(0, len(self.datas[test_class])), 2)

        test_image = transf(self.datas[test_class][image_indices[0]])

        images1[0] = transf(self.datas[test_class][image_indices[1]])

        available_classes = list(range(0, self.num_classes))
        available_classes.remove(test_class)

        different_classes = random.sample(range(0, len(available_classes)), self.way)

        for idx, img_idx in zip(different_classes, range(1, self.way+1)):
            images1[img_idx] = transf(random.choice(self.datas[idx]))

        images2 = torch.zeros(1, 1, self.image_width, self.image_height)
        images2[0] = test_image
        images2 = images2.expand(self.way+1, 1, self.image_width, self.image_height)
        return images1, images2

    def __len__(self):
        return self.times * self.way

    def __getitem__(self, index):
        idx = index % self.way
        label = None
        # generate image pair from same class
        if idx == 0:
            self.c1 = random.randint(0, self.num_classes - 1)
            self.img1 = random.choice(self.datas[self.c1])
            img2 = random.choice(self.datas[self.c1])
        # generate image pair from different class
        else:
            c2 = random.randint(0, self.num_classes - 1)
            while self.c1 == c2:
                c2 = random.randint(0, self.num_classes - 1)
            img2 = random.choice(self.datas[c2])

        if self.transform:
            img1 = self.transform(self.img1)
            img2 = self.transform(img2)
        return img1, img2


# test
if __name__=='__main__':
    omniglotTrain = OmniglotTrain('./images_background', 30000*8)
    print(omniglotTrain)