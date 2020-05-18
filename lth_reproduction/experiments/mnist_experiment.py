import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import gin

import sys
sys.path.append('../src')

from models import LeNetFC, LeNetConv
from train import train
from evaluate import test
from prepare_data import load_mnist

@gin.configurable
def main(
        model=LeNetFC,
        batch_size=64,
        test_batch_size=1000,
        epochs=3,
        lr=1.0,
        gamma=0.7,
        no_cuda=False,
        rand_seed=42,
        save_model=False,
        conv_layers=False
        ):
    """
    This is the main script which trains and tests the model

    Args:
        model
        batch_size (int): size of training mini-batch
        test_batch_size (int): size of testing batch
        epochs (int): num epochs
        lr (float): learning rate
        gamma (float): rate at which to adjust lr with scheduler
        no_cuda (bool): cuda or not
        rand_seed (int): random seed
        save_model (bool): whether to save pytorch model
        conv_layers (bool): whether to include convolutional layers in LeNet architecture or not
    """
    # view model
    print(model)

    # load data
    train_loader, test_loader, use_cuda = load_mnist(batch_size, test_batch_size, no_cuda, rand_seed)

    # setup device, model, optimizer, and lr scheduler
    device = torch.device('cuda' if use_cuda else 'cpu')
    print('device:', device)
    model = LeNetConv().to(device) if conv_layers else LeNetFC().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

    # run the training loop
    for epoch in range(1, epochs + 1):
        done = train(model, device, train_loader, test_loader, optimizer, epoch)

        # here we use the test function for validation
        test(model, device, test_loader)
        scheduler.step()

        if done:
            break

    if save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    gin.parse_config_file('../config/mnist_config.gin')
    main()
