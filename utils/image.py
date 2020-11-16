import matplotlib.pyplot as plt
import numpy as np
import torchvision


def visualize_dataset(dataiter, size):
    grid = [dataiter.next()[0] for i in range(size)]
    imshow(torchvision.utils.make_grid(grid, nrow=200, padding=2))


def imshow(img):
    img = img / 2 + 0.5  # denormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
