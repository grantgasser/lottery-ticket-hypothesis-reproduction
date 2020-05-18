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
        for data, target in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += model.loss(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)

    print('\nTest set: Average loss: {:.6f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return test_loss

def validate(model, device, test_loader, val_size, val_batch_size):
    """
    This function runs the testing script of the model, testing on a batch of test examples
    each time its called and printing the results

    Args:
        model (obj): which model to train
        device (torch.device): device to run on, cpu or whether to enable cuda
        test_loader (torch.utils.data.dataloader.DataLoader): dataloader object
        val_size (int): size of validation set (subset of test set)
        val_batch_size (int): size of batches for validation set
    """
    iters = val_size / val_batch_size
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += model.loss(output, target).item()  # sum up batch loss

            # only validating on subset of test_loader set
            if batch_idx == (iters - 1):
                val_loss /= iters
                break

    return val_loss
