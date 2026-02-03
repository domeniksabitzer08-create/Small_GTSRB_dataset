import builtins
from array import array

import torch
import torchvision
from numpy.ma.core import indices
from sympy.testing.pytest import warns

from torch.utils.data import Dataset, Subset

from torchvision import datasets, transforms
from torchvision.transforms import ToTensor

from random import randint
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm




def get_img_index_per_class(n_classes: int, data: torch.utils.data.Dataset):
    """Gets all indexes from all classes and returns them as an array"""
    img_idx_per_class = []
    for i in tqdm(range(n_classes)):
        indexe = []
        for j in range(len(data)):
            img, label = data[j]
            if label == i:
                indexe.append(j)
        img_idx_per_class.append(indexe)
    return img_idx_per_class

def get_number_of_imgs_per_class(img_idx_per_class: list):
    """Counts the number of images per class and returns them as an array"""
    n_img_per_class = []
    for i in range(len(img_idx_per_class)):
        n_img_per_class.append(len(img_idx_per_class[i]))
    return n_img_per_class



def get_indices(data_idxs , sub_classes: iter, max_imgs: int):
    indices = []
    for i in sub_classes:
        for j in range(max_imgs):
            rnd_idx = randint(0, len(data_idxs[i])-1)
            indices.append(data_idxs[i][rnd_idx])
    #indices = set(indices) # currently outcommented for debugging
    return indices

def get_max_imgs(sub_classes: iter, data_idxs: list,  balancing : bool ,n_max_imgs: int = None):
    data_idxs = list(data_idxs)
    max_imgs = []
    for i in sub_classes:
        max_imgs.append(len(data_idxs[i]))
    if balancing:
        return min(max_imgs)
    else:
        # Check if the classes have enough imgs for n_max_imgs
        if n_max_imgs is None:
            return min(max_imgs)
        elif n_max_imgs <= min(max_imgs):
            return n_max_imgs
        else:
            warns(f"Not enough images, n_max_imgs is too high! - using the highest possible number of images which is {min(max_imgs)}")
            return min(max_imgs)

def get_map_label(sub_classes: iter, data_idxs: iter):
    # create a list that index contains all elements to the label(index)
    map_label = []
    for i in sub_classes:
        map_label.append(data_idxs[i])
    return map_label



def make_subset(n_classes: int, sub_classes: iter, full_train_data : torch.utils.data.dataset, full_test_data: torch.utils.data.dataset, balancing: bool, n_max_imgs: int = None):
    # get the indexes and count of training and testing data
    train_img_idxs = get_img_index_per_class(n_classes, full_train_data)
    test_img_idxs = get_img_index_per_class(n_classes, full_test_data)
    # get the number of imgs per class for balancing
    train_img_count = get_number_of_imgs_per_class(train_img_idxs)
    test_img_count = get_number_of_imgs_per_class(test_img_idxs)
    # get the max imgs
    max_imgs = get_max_imgs(sub_classes, train_img_idxs, balancing, n_max_imgs)
    print(f"max images: {max_imgs}")
    # get indices
    indices = get_indices(train_img_idxs, sub_classes, max_imgs)
    print(f"Inicies: {indices}")
    return indices

class GTSRBSubset(Dataset):
    """A Subset of the GTSRB Dataset."""

    def __init__(self, base_dataset, indices, label_mapping):
        self.base_dataset = base_dataset
        self.indices = indices
        self.label_mapping = label_mapping

    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        # ich hab label und index vertauscht !!!
        image, original_label = self.base_dataset[original_idx]
        for i in range(len(self.label_mapping)):
            try:
                new_label = self.label_mapping[i].index(original_label)
            except ValueError:
                pass

        return image, new_label


