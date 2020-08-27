import os
import PIL
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from os import listdir
from os.path import join
from PIL import Image
from utils.transform import itensity_normalize
from torch.utils.data.dataset import Dataset


class ISIC2018_dataset(Dataset):
    def __init__(self, dataset_folder='/ISIC2018_Task1_npy_all',
                 folder='folder0', train_type='train', transform=None):
        self.transform = transform
        self.train_type = train_type
        self.folder_file = './Datasets/' + folder

        if self.train_type in ['train', 'validation', 'test']:
            # this is for cross validation
            with open(join(self.folder_file, self.folder_file.split('/')[-1] + '_' + self.train_type + '.list'),
                      'r') as f:
                self.image_list = f.readlines()
            self.image_list = [item.replace('\n', '') for item in self.image_list]
            self.folder = [join(dataset_folder, 'image', x) for x in self.image_list]
            self.mask = [join(dataset_folder, 'label', x.split('.')[0] + '_segmentation.npy') for x in self.image_list]
            # self.folder = sorted([join(dataset_folder, self.train_type, 'image', x) for x in
            #                       listdir(join(dataset_folder, self.train_type, 'image'))])
            # self.mask = sorted([join(dataset_folder, self.train_type, 'label', x) for x in
            #                     listdir(join(dataset_folder, self.train_type, 'label'))])
        else:
            print("Choosing type error, You have to choose the loading data type including: train, validation, test")

        assert len(self.folder) == len(self.mask)

    def __getitem__(self, item: int):
        image = np.load(self.folder[item])
        label = np.load(self.mask[item])

        sample = {'image': image, 'label': label}

        if self.transform is not None:
            # TODO: transformation to argument datasets
            sample = self.transform(sample, self.train_type)

        return sample['image'], sample['label']

    def __len__(self):
        return len(self.folder)

# a = ISIC2018_dataset()
