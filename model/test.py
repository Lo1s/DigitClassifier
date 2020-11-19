import torch
import torch.nn.functional as F

@DeprecationWarning
def batch_accuracy(xb, yb):
    preds = xb.sigmoid()
    correct = (preds > 0.5) == yb
    return correct.float().mean()


@DeprecationWarning
def validate_epoch(model, testloader):
    accs = [batch_accuracy(model(xb), yb) for xb, yb in testloader]
    return round(torch.stack(accs).mean().item(), 4)


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