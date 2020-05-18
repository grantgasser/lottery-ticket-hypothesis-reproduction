import torch
import gin

from evaluate import validate


@gin.configurable
def train(model, device, train_loader, test_loader, optimizer, epoch, batch_log_interval=10, val_size=256, val_batch_size=64):
    """
    This function runs the training script of the model

    Args:
        model (obj): which model to train
        device (torch.device): device to run on, cpu or whether to enable cuda
        train_loader (torch.utils.data.dataloader.DataLoader): dataloader object
        test_loader (torch.utils.data.dataloader.DataLoader): dataloader object for validation
        epoch (int): which epoch we're on
        optimizer (torch.optim obj): which optimizer to use
        batch_log_interval (int): how often to log results

    Returns
        done (bool): whether to stop training because of early stopping
    """
    patience = 20  # in terms of iterations (batches), not epochs
    no_improvement_count = 0
    min_delta = 0.01
    done = False

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
            val_loss = validate(model, device, test_loader, val_size, val_batch_size)
            prev_val_loss = val_loss
        else:
            prev_val_loss = val_loss
            val_loss = validate(model, device, test_loader, val_size, val_batch_size)

        # if no improvement on this batch
        if abs(prev_val_loss - val_loss) < min_delta:
            no_improvement_count += 1
        else:
            no_improvement_count = 0

        if no_improvement_count == patience:
            print('Early Stopping Triggered. Done Training. val_loss = {}, prev_val_loss = {}'.format(val_loss, prev_val_loss))
            done = True
            break

        if batch_idx % batch_log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    return done
