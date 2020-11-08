from __future__ import print_function
import torch

from utils.data import download_file, URLs

if __name__ == '__main__':
    datasets = URLs.MNIST_DATASET
    print(datasets)

    for dataset in datasets:
        print(dataset)
        filepath = download_file(dataset)
        print(filepath)
