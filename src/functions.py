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