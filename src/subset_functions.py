import builtins
from array import array

import torch
import torchvision

from torch.utils.data import Dataset, Subset

from torchvision import datasets, transforms
from torchvision.transforms import ToTensor

from random import randint
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

import functions
from src.Main import img_idx_per_class


def get_img_index_per_class(n_classes: iter, data: torch.utils.data.Dataset):
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



def make_subset(sub_classes: list, full_train_data : torch.utils.data.dataset, full_test_data: torch.utils.data.dataset, balancing: bool):
    print("Still working on it")

