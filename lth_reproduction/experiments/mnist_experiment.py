import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import gin

from 

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