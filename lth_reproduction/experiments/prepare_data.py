import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize


def load_mnist(train_batch_size=64, val_batch_size=1000, no_cuda=False, rand_seed=42):
    """
    Loads mnist dataset from Torch and stores for training and testing.

    Args:
        train_batch_size (int): size of training mini-batch
        val_batch_size (int): size of validation batch
        no_cuda (bool): whether to use cuda or not
        rand_seed (int): random seed

    Returns:
        train_loader (torch.utils.data.DataLoader): object containing training data
        val_loader (torch.utils.data.DataLoader): object containing validation data
        use_cuda (bool): whether to use cuda
    """
    # logistics
    use_cuda = not no_cuda and torch.cuda.is_available()
    print('cuda available?:', torch.cuda.is_available())
    print('use_cuda:', use_cuda)
    torch.manual_seed(rand_seed)
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # load train and validation data
    data_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    train_loader = DataLoader(
        MNIST(root='../data', train=True, download=True, transform=data_transform),
        batch_size=train_batch_size, shuffle=True, **kwargs
    )

    val_loader = DataLoader(
        MNIST(root='../data', train=False, download=False, transform=data_transform),
        batch_size=val_batch_size, shuffle=False, **kwargs
    )

    return train_loader, val_loader, use_cuda
