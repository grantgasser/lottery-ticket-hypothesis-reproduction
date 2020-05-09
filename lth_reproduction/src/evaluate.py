import torch
import torch.nn.functional as F


def test(model, device, test_loader):
    """
    This function runs the testing script of the model, testing on a batch of test examples
    each time its called and printing the results

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
