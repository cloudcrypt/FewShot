import torch
from torch.utils.data import Dataset, DataLoader
import os
import glob as glob
from numpy.random import choice as npc
import numpy as np
import time
import random
import torchvision.datasets as dset
from PIL import Image
import errno

class PoseDatasetLoader:
    urls = [
        'https://drive.google.com/a/robotics.utias.utoronto.ca/uc?export=download&confirm=XaSO&id=1_21pFglIueUE0f13AtHFSdyxugEo9K_f'
    ]
    raw_folder = 'raw'
    processed_folder = 'processed'

    def __init__(self, path):
        self.dataset_path = path

    def download_if_necessary(self):
        if not self._check_exists():
            self.download()

    def _check_exists(self):
        return os.path.exists(os.path.join(self.dataset_path, self.processed_folder, "test")) and \
               os.path.exists(os.path.join(self.dataset_path, self.processed_folder, "train"))

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


class PoseDatasetTrain(Dataset):

    def __init__(self, dataPath, transform=None):
        super(PoseDatasetTrain, self).__init__()

        #loader = PoseDatasetLoader(dataPath)
        #loader.download_if_necessary()

        np.random.seed(0)
        # self.dataset = dataset
        self.transform = transform
        self.datas, self.num_classes = self.loadToMem(dataPath)

    def loadToMem(self, dataPath):
        print("begin loading training dataset to memory")
        datas = {}
        agrees = [0, 90, 180, 270]
        idx = 0
        imgNames = glob.glob(dataPath + "\\train\\gen\\" + "*.png", recursive=True)
        for imgName in imgNames:
            datas[idx]=[]
            datas[idx].append(Image.open(imgName).convert('L'))
            datas[idx].append(Image.open(imgName.replace("gen","real").replace(".png","_R.png")).convert('L'))
            idx += 1
        print("finish loading training dataset to memory")
        return datas, idx

    def __len__(self):
        return  21000000

    def __getitem__(self, index):
        # image1 = random.choice(self.dataset.imgs)
        label = None
        img1 = None
        img2 = None
        # get image from same class
        if index % 2 == 1:
            label = 1.0
            idx1 = random.randint(0, self.num_classes - 1)
            image1 = self.datas[idx1][0]
            image2 = self.datas[idx1][1]
        # get image from different class
        else:
            label = 0.0
            idx1 = random.randint(0, self.num_classes - 1)
            idx2 = random.randint(0, self.num_classes - 1)
            while idx1 == idx2:
                idx2 = random.randint(0, self.num_classes - 1)
            image1 = self.datas[idx1][0]
            image2 = self.datas[idx2][1]

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        return image1, image2, torch.from_numpy(np.array([label], dtype=np.float32))


class PoseDatasetTest(Dataset):

    def __init__(self, dataPath, transform=None, times=200, way=20):
        np.random.seed(1)
        super(PoseDatasetTest, self).__init__()
        loader = OmniglotLoader(dataPath)
        loader.download_if_necessary()

        self.transform = transform
        self.times = times
        self.way = way
        self.img1 = None
        self.c1 = None
        self.datas, self.num_classes = self.loadToMem(dataPath)

    def loadToMem(self, dataPath):
        print("begin loading test dataset to memory")
        datas = {}
        idx = 0
        imgNames = glob.glob(dataPath + "\\test\\gen\\" + "*.png", recursive=True)
        for imgName in imgNames:
            datas[idx] = []
            datas[idx].append(Image.open(imgName).convert('L'))
            datas[idx].append(Image.open(imgName.replace("gen", "real").replace(".png", "_R.png")).convert('L'))
            idx += 1
        print("finish loading test dataset to memory")
        return datas, idx

    def __len__(self):
        return self.times * self.way

    def __getitem__(self, index):
        idx = index % self.way
        label = None
        # generate image pair from same class
        if idx == 0:
            self.c1 = random.randint(0, self.num_classes - 1)
            self.img1 = self.datas[self.c1][0]
            img2 = self.datas[self.c1][1]
        # generate image pair from different class
        else:
            c2 = random.randint(0, self.num_classes - 1)
            while self.c1 == c2:
                c2 = random.randint(0, self.num_classes - 1)
            img2 = self.datas[c2][1]

        if self.transform:
            img1 = self.transform(self.img1)
            img2 = self.transform(img2)
        return img1, img2


# test
if __name__=='__main__':
    omniglotTrain = PoseDatasetTrain('D:/Downloads/bpM6x25_C/bpM6x25_C', 30000*8)
    print(omniglotTrain.num_classes)