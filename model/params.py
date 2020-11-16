import torch


def init_params(size, std=1.0): return (torch.randn(size)*std).requires_grad_()
