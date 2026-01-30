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
from src.functions import plot_img

# Get data
train_data = torchvision.datasets.GTSRB(root="data", split = "train", transform=ToTensor() ,download=True)
test_data = torchvision.datasets.GTSRB(root="data", split = "test", transform=ToTensor() ,download=True)

# Creating a subset for faster testing - train and test data have the same lenght
subset_lenght = 500
indicies = range(0, subset_lenght)
train_data_sub = torch.utils.data.Subset(train_data, indicies)
test_data_sub = torch.utils.data.Subset(test_data, indicies)

print(f"Train data lenght: {len(train_data)} | Test data lenght: {len(test_data)}")
print(f"Train data subset lenght: {len(train_data_sub)} | Test data subset lenght: {len(test_data_sub)}")

# Classes: hard coded
classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41]
# Classes: soft coded
# classes = functions.get_classes(test_data)


# Get number of images of each class


img_idx_per_class = []

n_classes = 42 # Can be smaller for faster testing

for i in tqdm(range(n_classes)):
    indexe = []
    for j in range(len(test_data)):
        img, label = test_data[j]
        if label == i:
            indexe.append(j)
    img_idx_per_class.append(indexe)

print(img_idx_per_class)
print(f"Shape: {len(img_idx_per_class)}")


# Print the lenght of each class and plot a example image
#for i in range(len(img_idx_per_class)):
    #print(f"Number of images in class {i+1}: {len(img_idx_per_class[i])}")
    #plot_img(test_data_sub, img_idx_per_class[i][0])

# Getting the number of images per class
n_img_per_class = []
for i in range(len(img_idx_per_class)):
    n_img_per_class.append(len(img_idx_per_class[i]))

print(f"Number of images per class: {len(classes)}")

functions.plot_bar_chart(classes,n_img_per_class , "Classes", "Number of img")

