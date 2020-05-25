import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.utils.prune as prune
import numpy as np
import gin

import sys
sys.path.append('../src')

from models import Conv2
from train import train
from evaluate import test
from prepare_data import load_cifar10
from pruning import Model

@gin.configurable
def main(
        model=Conv2,
        batch_size=64,
        train_size=20000,
        test_batch_size=1000,
        epochs=3,
        lr=1.0,
        gamma=0.7,
        no_cuda=False,
        rand_seed=42,
        save_model=False,
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
    """
    # view model
    print(model)

    # load data
    train_loader, val_loader, test_loader, use_cuda = load_cifar10(batch_size, train_size, test_batch_size, no_cuda, rand_seed)
    print(len(train_loader.dataset))
    # setup device, model, optimizer, and lr scheduler
    device = torch.device('cuda' if use_cuda else 'cpu')
    print('device:', device)
    model = Conv2().to(device)

    # our pruning
    # prune = BasePruning(model)
    # prune.store_init_weights()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

    # run the training loop
    for epoch in range(1, epochs + 1):
        stop, stopping_iteration = train(model, device, train_loader, val_loader, test_loader, optimizer, epoch)

        # test after each epoch
        scheduler.step()

        if stop:
            print('Stopped at overall iteration {}\n'.format(stopping_iteration+ ((len(train_loader.dataset)/batch_size) * (epoch-1))))
            break

    if save_model:
        torch.save(model.state_dict(), "cifar10_conv.pt")

    print('\nPruning...\n')
    prune_model = Model(model)
    prune_model.prune()

    # now predict w/ pruned network
    test(model, device, test_loader)


if __name__ == '__main__':
    gin.parse_config_file('../config/cifar10_config.gin')
    main()