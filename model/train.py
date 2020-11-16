from model.loss import mnist_loss
import torch

def calc_grad(x, y, weights, bias, model):
    preds = model(x, weights, bias)
    loss = mnist_loss(preds, y)
    loss.backward()


def train_epoch(model, dataloader, lr, params):
    weights, bias = params

    for xb, yb in dataloader:
        calc_grad(xb, yb, weights, bias, model)
        for p in params:
            p.data -= p.grad * lr
            p.grad.zero_()


def batch_accuracy(xb, yb):
    preds = xb.sigmoid()
    correct = (preds > 0.5) == yb
    return correct.float().mean()


def validate_epoch(model, weights, bias, testloader):
    accs = [batch_accuracy(model(xb, weights, bias), yb) for xb, yb in testloader]
    return round(torch.stack(accs).mean().item(), 4)