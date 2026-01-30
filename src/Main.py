import torch
import torchvision
from matplotlib.pyplot import plot_date

from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import ToTensor


def plot_img(data, index):
    img, label = data[index]
    print(type(img))
    plt.imshow(img.permute(1,2,0))


# Get data

train_data = torchvision.datasets.GTSRB(root="data", split = "train", transform=ToTensor() ,download=True)
test_data = torchvision.datasets.GTSRB(root="data", split = "test", transform=ToTensor() ,download=True)

# Plot img
plot_img(train_data, 1)




