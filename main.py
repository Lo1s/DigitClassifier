from __future__ import print_function
import torch
import os
import ipyplot
import webbrowser
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torch.utils.data import TensorDataset, DataLoader
from PIL import Image
from utils.data import download_file, URLs, extract_file, get_mnist_dataset
from utils.image import imshow, visualize_dataset

if __name__ == '__main__':
    data = get_mnist_dataset(print_progress=False)
    images = data['train']['images']
    labels = data['train']['labels']
    torch_images = torch.Tensor(images)
    torch_labels = torch.Tensor(labels)

    trainset = TensorDataset(torch_images, torch_labels)
    trainloader = DataLoader(trainset)

    dataiter = iter(trainloader)
    visualize_dataset(dataiter, 10000)
