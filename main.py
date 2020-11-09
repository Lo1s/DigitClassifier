from __future__ import print_function
import torch
import os

from utils.data import download_file, URLs, extract_file, get_mnist_dataset

if __name__ == '__main__':
    data = get_mnist_dataset(print_progress=False)
    print(data['train']['images'][0])
    print(data['train']['labels'])
