import torch
import torch.nn as nn
import torch.nn.functional as F
import gin


@gin.configurable
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

        self.loss = nn.CrossEntropyLoss()

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


@gin.configurable
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

        self.loss = nn.CrossEntropyLoss()

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
class Conv2(nn.Module):
    """
    Conv-2 Architecture

    # input: torch.Size([batch, 3, 32, 32])
    """
    def __init__(self):
        super(Conv2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        # From LTH: FC Layers 256, 256, 10
        self.fc1 = nn.Linear(12544, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)

        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        # Convolutions
        x = self.conv1(x)  # torch.Size([batch, 64, 30, 30])
        x = F.relu(x)
        x = self.conv2(x)  # torch.Size([batch, 64, 28, 28])
        x = F.relu(x)
        x = self.pool1(x)  # torch.Size([batch, 64, 14, 14])
        x = torch.flatten(x, 1)  # torch.Size([batch, 12544])

        # Full connection
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output

