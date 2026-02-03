from array import array

import torch
import torchvision
from sympy.core.random import shuffle

from torch.utils.data import Dataset, Subset
from torch.utils.hipify.hipify_python import mapping

from torchvision import datasets, transforms
from torchvision.transforms import ToTensor

from random import randint
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

import functions
import subset_functions
from subset_functions import get_img_index_per_class


# Get data
train_data = torchvision.datasets.GTSRB(root="data", split = "train", transform=ToTensor() ,download=True)
test_data = torchvision.datasets.GTSRB(root="data", split = "test", transform=ToTensor() ,download=True)

# Creating a subset for faster testing - train and test data have the same lenght
subset_lenght = 1000
indicies = range(0, subset_lenght)
train_data_sub = torch.utils.data.Subset(train_data, indicies)
test_data_sub = torch.utils.data.Subset(test_data, indicies)

print(f"Train data lenght: {len(train_data)} | Test data lenght: {len(test_data)}")
print(f"Train data subset lenght: {len(train_data_sub)} | Test data subset lenght: {len(test_data_sub)}")

# Classes: hard coded
classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41]
# Classes: soft coded
# classes = functions.get_classes(test_data)

n_classes = 42 # Can be smaller for faster testing

sub_classes = [22,33,34]


subset_indices = subset_functions.make_subset(n_classes ,sub_classes, test_data_sub, test_data_sub, balancing=True, n_max_imgs=None)
label_mapping = subset_functions.get_map_label(sub_classes, subset_functions.get_img_index_per_class(n_classes,test_data_sub))

print(f"label after mapping: {label_mapping}")

train_data_sub_me = subset_functions.GTSRBSubset(test_data_sub, subset_indices,label_mapping)



# try indexing and plotting
#functions.plot_img(train_data_sub_me, 8)
# plot all images
print(f"len of my subset: {len(train_data_sub_me)}")
for i in range(len(train_data_sub_me)):
    print(i)
    functions.plot_img(train_data_sub_me, i)







# plot a char of the numbers of images per class
#functions.plot_bar_chart(classes,subset_functions.get_number_of_imgs_per_class(subset_functions.get_img_index_per_class(n_classes, train_data_sub)) , "Classes", "Number of img")
# plot one image per class
#functions.plot_img_per_class(test_data, classes)

