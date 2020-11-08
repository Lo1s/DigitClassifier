import requests
import os
import re
import validators
from pathlib import Path


class URLs:
    PROJECT_PATH = Path.cwd()
    DATA_DIR = 'data'
    DATA_PATH = PROJECT_PATH / DATA_DIR
    MNIST_TRAIN_IMAGES = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
    MNIST_TRAIN_LABELS = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
    MNIST_TEST_IMAGES = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
    MNIST_TEST_LABELS = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
    MNIST_DATASET = [MNIST_TRAIN_IMAGES, MNIST_TRAIN_LABELS, MNIST_TEST_IMAGES, MNIST_TEST_LABELS]


def download_file(url, path=URLs.DATA_PATH):
    if not os.path.exists(path):
        path = URLs.DATA_PATH
    if not validators.url(url):
        print(f'URL={url} is not valid')
        return

    if not os.path.exists(path):
        os.mkdir(path)

    response = requests.get(url)
    filepath = path / os.path.basename(url)

    with filepath.open('wb') as f:
        f.write(response.content)

    return filepath
