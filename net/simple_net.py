import torch.nn as nn


def simple_net():
    return nn.Sequential(
        nn.Linear(28 * 28, 1),
        nn.ReLU(),
        nn.Linear(1, 256)
    )

