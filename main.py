from __future__ import print_function

import time

import torch
import torch.optim as optim
import torchvision
from torch.utils.data import TensorDataset

from model.train import train, test
from net.Net import Net
from utils.data import URLs
from utils.plot import plot_perf

if __name__ == '__main__':
    # data = get_mnist_dataset(print_progress=False)
    # train_images = data['train']['images']
    # train_labels = data['train']['labels']
    # test_images = data['test']['images']
    # test_labels = data['test']['labels']

    # visualize_dataset(train_images, train_labels, 10000)

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(URLs.DATA_PATH, train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=64, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(URLs.DATA_PATH, train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=1000, shuffle=True)

    lr = 0.01
    epochs = 3
    model = Net()
    # model_state_dict = torch.load(str(URLs.RESULTS_PATH / 'model.pth'))
    # model.load_state_dict(model_state_dict)
    opt = optim.SGD(model.parameters(), lr, momentum=0.5)
    # opt_state_dict = torch.load(str(URLs.RESULTS_PATH / 'optimizer.pth'))
    # opt.load_state_dict(opt_state_dict)
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i * len(train_loader.dataset) for i in range(epochs + 1)]
    device = torch.device("cpu")

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        model.cuda()

    start_time = time.time()
    test(model, test_loader, device, test_losses)
    for epoch in range(1, epochs + 1):
        train(epoch, model, opt, train_loader, device, train_losses, train_counter, 10)
        test(model, test_loader, device, test_losses)
    print(f'Running time: {time.strftime("%M:%S", time.gmtime(time.time() - start_time))} sec')
    plot_perf(train_counter, train_losses, test_counter, test_losses)
