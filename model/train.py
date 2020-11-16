from model.BasicOptim import BasicOptim
from model.loss import mnist_loss
import torch
import torch.nn as nn


def calc_grad(x, y, model):
    preds = model(x)
    loss = mnist_loss(preds, y)
    loss.backward()


def train_epoch(model, dataloader, lr):
    opt = BasicOptim(model.parameters(), lr)
    for xb, yb in dataloader:
        calc_grad(xb, yb, model)
        opt.step()
        opt.zero_grad()


def batch_accuracy(xb, yb):
    preds = xb.sigmoid()
    correct = (preds > 0.5) == yb
    return correct.float().mean()


def validate_epoch(model, testloader):
    accs = [batch_accuracy(model(xb), yb) for xb, yb in testloader]
    return round(torch.stack(accs).mean().item(), 4)


def train_model(model, dataloader, testloader, lr, epochs):
    for i in range(epochs):
        train_epoch(model, dataloader, lr)
        print(f'Epoch {i}: {validate_epoch(model, testloader)}')