from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import gin


class LeNetFC(nn.Module):
    """
    LeNet Architecture from LTH Paper, no convolutional layers

    # input: torch.Size([64, 1, 28, 28])
    """
    def __init__(self):
        super(LeNetFC, self).__init__()
        self.fc1 = nn.Linear(784, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)

        # Full connection
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output

class LeNetConv(nn.Module):
    """
    LeNet Convolutional Architecture

    # input: torch.Size([64, 1, 28, 28])
    """
    def __init__(self):
        super(LeNetConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5))
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5))

        # From LTH: FC Layers 300, 100, 10
        self.fc1 = nn.Linear(16*4*4, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
    	# Convolutions
        x = self.conv1(x)  # torch.Size([64, 6, 24, 24])
        x = F.relu(x)
        x = self.pool1(x)  # torch.Size([64, 6, 12, 12])
        x = self.conv2(x)  # torch.Size([64, 16, 8, 8])
        x = F.relu(x)
        x = self.pool1(x)  # torch.Size([64, 16, 4, 4])
        x = torch.flatten(x, 1)  # torch.Size([64, 256])

        # Full connection
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output


@gin.configurable
def train(model, device, train_loader, optimizer, epoch, batch_log_interval=10):
    """
    This function runs the training script of the model

    Args:
        model (obj): which model to train
        device (torch.device): device to run on, cpu or whether to enable cuda
        train_loader (torch.utils.data.dataloader.DataLoader): dataloader object
        epoch (int): which epoch we're on
        optimizer (torch.optim obj): which optimizer to use
        batch_log_interval (int): how often to log results
    """
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % batch_log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    """
    This function runs the testing script of the model

    Args:
        model (obj): which model to train
        device (torch.device): device to run on, cpu or whether to enable cuda
        test_loader (torch.utils.data.dataloader.DataLoader): dataloader object
    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


@gin.configurable
def main(
        batch_size=64,
        test_batch_size=1000,
        epochs=5,
        lr=1.0,
        gamma=0.7,
        no_cuda=False,
        rand_seed=1,
        save_model=False,
        conv_layers=False
        ):
    """
    This is the main script which trains and tests the model

    Args:
        batch_size (int)
        test_batch_size (int)
        epochs (int): num epochs
        lr (float): learning rate
        gamma (float): rate at which to adjust lr with scheduler
        no_cuda (bool): cuda or not
        rand_seed (int): random seed
        save_model (bool): whether to save pytorch model
        conv_layers (bool): whether to include convolutional layers in LeNet architecture or not
    """
    use_cuda = not no_cuda and torch.cuda.is_available()

    torch.manual_seed(rand_seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=test_batch_size, shuffle=True, **kwargs)

    model = LeNetConv().to(device) if conv_layers else LeNetFC().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    gin.parse_config_file('mnist-config.gin')
    main()

