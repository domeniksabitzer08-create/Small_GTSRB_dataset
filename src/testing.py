import torch
import torchvision

from torchvision.transforms import ToTensor

import functions
from GTSRB_SubsetMaker import subset_functions

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

sub_classes = [22,33,34,12,18]
classes_new = range(0, len(sub_classes))

#subset_indices, train_img_idxs, test_img_idxs = subset_functions.make_subset(n_classes ,sub_classes, test_data_sub, test_data_sub, balancing=True, n_max_imgs=None)
#label_mapping = subset_functions.get_map_label(sub_classes, train_img_idxs)

#print(f"label after mapping: {label_mapping}")

train_data_sub_me = subset_functions.GTSRBSubset(test_data_sub, n_classes, sub_classes, balancing=True, n_max_imgs=None)

print(f"len of my subset: {len(train_data_sub_me)}")
# Plot one img per class
functions.plot_img_per_class(train_data_sub_me, classes_new, subset_functions.get_img_index_per_class(len(classes_new), train_data_sub_me))




# functions.plot_img_per_class(test_data_sub, classes, train_img_idxs)
# plot a char of the numbers of images per class
#functions.plot_bar_chart(classes,subset_functions.get_number_of_imgs_per_class(subset_functions.get_img_index_per_class(n_classes, train_data_sub)) , "Classes", "Number of img")
# plot one image per class
#functions.plot_img_per_class(test_data, classes)

