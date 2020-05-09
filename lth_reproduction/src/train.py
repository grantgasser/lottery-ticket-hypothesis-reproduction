import torch
import torch.nn.functional as F
import gin


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
