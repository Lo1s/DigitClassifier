import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch.utils.data import TensorDataset, DataLoader


def visualize_dataset(images, labels, size):
    torch_images = torch.Tensor(images)
    torch_labels = torch.Tensor(labels)

    trainset = TensorDataset(torch_images, torch_labels)
    trainloader = DataLoader(trainset)

    dataiter = iter(trainloader)
    grid = [dataiter.next()[0] for _ in range(size)]
    imshow(torchvision.utils.make_grid(grid, nrow=200, padding=2))


def visualize_dataset_tensorboard(trainloader, writer):
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    grid = torchvision.utils.make_grid(images)

    writer.add_image('mnist_images', grid)


def imshow(img):
    img = img / 2 + 0.5  # denormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
