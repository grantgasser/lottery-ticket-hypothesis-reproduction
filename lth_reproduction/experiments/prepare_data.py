import torch
from torch.utils.data import SubsetRandomSampler
from torchvision import datasets, transforms
import numpy as np


def load_mnist(batch_size=64, valid_size=500, no_cuda=False, rand_seed=42, shuffle=True):
    """
    Loads mnist dataset from Torch and stores for training and testing.

    Args:
        batch_size (int): size of training mini-batch
        test_batch_size (int): size of testing batch
        no_cuda (bool): whether to use cuda or not
        rand_seed (int): random seed

    Returns:
        train_loader (torch.utils.data.DataLoader): object containing training data
        test_loader (torch.utils.data.DataLoader): object containing test data
        use_cuda (bool): whether to use cuda

    TODO: change validation set to be 5000 of the original 60000 train samples (thus leaving 55000 train samples)
    """
    # logistics
    use_cuda = not no_cuda and torch.cuda.is_available()
    print('cuda avail?:', torch.cuda.is_available())
    print('use_cuda:', use_cuda)
    torch.manual_seed(rand_seed)
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # load train/valid data
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_data = datasets.MNIST(
        root='../data', train=True,
        download=True, transform=tfm
    )

    val_data = datasets.MNIST(
        root='../data', train=True,
        download=True, transform=tfm
    )

    num_train = len(train_data)
    indices = list(range(num_train))
    valid_size = 500

    if shuffle:
        np.random.seed(rand_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[valid_size:], indices[:valid_size]
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size,
        sampler=train_sampler, **kwargs
    )
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=batch_size,
        sampler=val_sampler, **kwargs
    )

    # load test data
    test_data = datasets.MNIST(
        root='../data', train=False,
        download=True, transform=tfm
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size,
        shuffle=True, **kwargs
    )

    return train_loader, val_loader, test_loader, use_cuda
