import os

import torch
import torch.nn.functional as F

from model.BasicOptim import BasicOptim
from utils.data import URLs


def calc_grad(x, y, model):
    preds = model(x)
    loss = F.nll_loss(preds, y)
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


def train(epoch, model, opt, train_loader, device, train_losses, train_counter, log_interval):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        opt.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        opt.step()
        if batch_idx % log_interval == 0:
            print(f'Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item()}')
            train_losses.append(loss.item())
            train_counter.append(batch_idx * 64 + ((epoch - 1) * len(train_loader.dataset)))

            if not os.path.exists(URLs.RESULTS_PATH):
                os.makedirs(URLs.RESULTS_PATH)
            torch.save(model.state_dict(), str(URLs.RESULTS_PATH / 'model.pth'))
            torch.save(opt.state_dict(), str(URLs.RESULTS_PATH / 'optimizer.pth'))


def test(model, test_loader, device, test_losses):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print(f'Test set: Avg. loss: {test_loss}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({100. * correct / len(test_loader.dataset)})')

