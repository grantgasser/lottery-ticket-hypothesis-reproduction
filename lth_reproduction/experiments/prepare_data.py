import torch
from torchvision import datasets, transforms


def load_mnist(batch_size=64, test_batch_size=1000, no_cuda=False, rand_seed=42):
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
    """
    # logistics
    use_cuda = not no_cuda and torch.cuda.is_available()
    torch.manual_seed(rand_seed)
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # load train and test data
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

    return train_loader, test_loader, use_cuda
