import torch
import gin

from evaluate import validate, test


@gin.configurable
def train(
        model,
        device,
        train_loader,
        val_loader,
        test_loader,
        optimizer,
        epoch,
        batch_log_interval=10,
        patience=20,
        min_delta=0.003):
    """
    This function runs the training script of the model

    Args:
        model (obj): which model to train
        device (torch.device): device to run on, cpu or whether to enable cuda
        train_loader (torch.utils.data.dataloader.DataLoader): dataloader object
        val_loader (torch.utils.data.dataloader.DataLoader): dataloader object for validation
        test_loader (torch.utils.data.dataloader.DataLoader): dataloader object for testing at
            would-be early stopping iteration
        epoch (int): which epoch we're on
        optimizer (torch.optim obj): which optimizer to use
        batch_log_interval (int): how often to log results
        patience (int): how many iterations/batches (not epochs) we will tolerate a val_loss improvement < min_delta
        min_delta (float): early stopping threshold; if val_loss < min_delta for patience # of iterations, we consider
            early stopping to have occurred

    Returns
        stop (bool): whether early stopping would have occurred
    """
    print('min_delta:', min_delta)
    no_improvement_count = 0
    stop = False

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = model.loss(output, target)
        loss.backward()
        optimizer.step()

        # for each "iteration" (assuming that means batch), get validation loss for early stopping
        if batch_idx == 0:
            val_loss = validate(model, device, val_loader)
            prev_val_loss = val_loss
        else:
            prev_val_loss = val_loss
            val_loss = validate(model, device, val_loader)

        # if no improvement on this batch
        if abs(prev_val_loss - val_loss) < min_delta:
            no_improvement_count += 1
        else:
            no_improvement_count = 0

        # trigger early stopping
        if no_improvement_count == patience:
            print('Early Stopping Triggered. Done Training. val_loss = {:.6f}, prev_val_loss = {:.6f}'.format(val_loss, prev_val_loss))
            test(model, device, test_loader)
            stop = True
            break

        if batch_idx % batch_log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    return stop
