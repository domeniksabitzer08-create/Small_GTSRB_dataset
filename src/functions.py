import torch
import torchvision
from matplotlib.pyplot import plot_date

from torchvision import datasets, transforms
from torchvision.transforms import ToTensor

from random import randint
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm





def plot_img(data, index):
    img, label = data[index]
    plt.imshow(img.permute(1,2,0))
    plt.axis(False)
    plt.title(label)
    plt.show()

def plot_bar_chart(x_labels, y_labels, x_name, y_name):
    plt.figure(1, figsize=(18, 8))
    plt.bar(x_labels, y_labels)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.xticks(range(0,len(x_labels)))

    plt.show()

def get_classes(data):
    labels = []
    for i in tqdm(range((len(data)))):
        img, label = data[i]
        labels.append(label)
    classes = set(labels)
    return classes

def plot_img_per_class(data, classes):
    for i in classes:
        img, label = data[i]
        plt.imshow(img.permute(1,2,0))
        plt.axis(False)
        plt.title(classes[i])
        plt.show()