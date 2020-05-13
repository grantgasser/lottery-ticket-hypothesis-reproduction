import sys
sys.path.append('../src')
import gin
import logging

import torch
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, RunningAverage
from ignite.contrib.handlers import ProgressBar

from models import LeNetFC, LeNetConv
from train import train
from evaluate import test
from prepare_data import load_mnist

@gin.configurable
def run(
        model=LeNetFC,
        train_batch_size=64,
        val_batch_size=1000,
        epochs=3,
        lr=1.0,
        gamma=0.7,
        no_cuda=False,
        rand_seed=42,
        save_model=False,
        display_gpu_info=False
        ):
    """
    This is the main script which trains and tests the model

    Args:
        model (subclass of nn.Module): which model to train
        train_batch_size (int): size of training mini-batch
        val_batch_size (int): size of testing batch
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
    train_loader, val_loader, use_cuda = load_mnist(train_batch_size, val_batch_size, no_cuda, rand_seed)

    # setup device, model, optimizer, and lr scheduler
    device = torch.device('cuda' if use_cuda else 'cpu')
    print('device:', device)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    #scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    loss_fn = nn.CrossEntropyLoss()

    # ignite engines
    # following tutorial: https://github.com/pytorch/ignite/tree/master/examples/contrib/mnist
    trainer = create_supervised_trainer(model, optimizer, loss_fn, device=device)
    evaluator = create_supervised_evaluator(
        model, metrics={'Accuracy': Accuracy(), 'Loss': Loss(loss_fn)}, device=device
    )

    if display_gpu_info:
        from ignite.contrib.metrics import GpuInfo

        GpuInfo().attach(trainer, name="gpu")

    pbar = ProgressBar(persist=True)
    pbar.attach(trainer, metric_names="all")

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['Accuracy']
        avg_loss = metrics['Loss']
        pbar.log_message(
            "Training Results - Epoch: {}  Avg accuracy: {:.2f}% Avg loss: {:.4f}".format(
                engine.state.epoch, avg_accuracy*100, avg_loss
            )
        )

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['Accuracy']
        avg_loss = metrics['Loss']
        pbar.log_message(
            "Validation Results - Epoch: {}  Avg accuracy: {:.2f}% Avg loss: {:.4f}".format(
                engine.state.epoch, avg_accuracy*100, avg_loss
            )
        )

        pbar.n = pbar.last_print_n = 0

    trainer.run(train_loader, max_epochs=epochs)


if __name__ == '__main__':
    gin.parse_config_file('../config/mnist_config.gin')
    run()
