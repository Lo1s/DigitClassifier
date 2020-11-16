from __future__ import print_function

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from model.train import train_model
from utils.data import get_mnist_dataset

if __name__ == '__main__':
    data = get_mnist_dataset(print_progress=False)
    train_images = data['train']['images']
    train_labels = data['train']['labels']
    test_images = data['test']['images']
    test_labels = data['test']['labels']

    # visualize_dataset(train_images, train_labels, 10000)

    torch_train_images = torch.Tensor(train_images).view(-1, 28*28)
    torch_train_labels = torch.Tensor(train_labels).unsqueeze(1)
    torch_test_images = torch.Tensor(test_images).view(-1, 28*28)
    torch_test_labels = torch.Tensor(test_labels).unsqueeze(1)

    trainset = TensorDataset(torch_train_images, torch_train_labels)
    trainloader = DataLoader(trainset, batch_size=256)

    testset = TensorDataset(torch_test_images, torch_test_labels)
    testloader = DataLoader(testset, batch_size=256)

    dataiter = iter(trainloader)

    # xb, yb = dataiter.next()
    # print(xb.shape, yb.shape)
    #
    # batch = xb[:4]
    # print(batch.shape)
    #
    # preds = linear1(batch, weights, bias)
    # print(preds)
    #
    # loss = mnist_loss(preds, yb[:4])
    # print(loss)
    #
    # loss.backward()
    # print(weights.grad.shape, weights.grad.mean(), bias.grad)
    #
    # calc_grad(batch, yb[:4], weights, bias, linear1)
    # print(weights.grad.mean(), bias.grad)
    #
    # weights.grad.zero_()
    # bias.grad.zero_()
    #
    # print(batch_accuracy(batch, yb[:4]))
    # print('before 1 epoch: ', validate_epoch(linear1, weights, bias, testloader))

    # train_epoch(linear1, trainloader, lr, params)
    # print('after 1 epoch: ', validate_epoch(linear1, weights, bias, testloader))

    lr = 1
    linear_model = nn.Linear(28*28, 1)
    train_model(linear_model, trainloader, testloader, lr, 20)
