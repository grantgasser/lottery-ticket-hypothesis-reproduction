import sys
sys.path.append('../src')
import gin
import logging

import torch
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.contrib.handlers.tensorboard_logger import *

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
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    loss_fn = nn.CrossEntropyLoss()

    # engines
    trainer = create_supervised_trainer(model, optimizer, loss_fn, device=device)
    metrics = {'Accuracy': Accuracy(), 'Loss': Loss(loss_fn)}
    train_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)
    val_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)

    # # run the training loop
    # for epoch in range(1, epochs + 1):
    #     train(model, device, train_loader, optimizer, epoch)
    #
    #     # here we use the test function for validation
    #     test(model, loss_fn, val_loader, device)
    #     scheduler.step()
    #
    # if save_model:
    #     torch.save(model.state_dict(), "mnist_cnn.pt")

    @trainer.on(Events.EPOCH_COMPLETED)
    def compute_metrics(engine):
        train_evaluator.run(train_loader)
        val_evaluator.run(val_loader)

    tb_logger = TensorboardLogger(log_dir='logs/tensorboard_logs')

    tb_logger.attach(
        trainer,
        log_handler=OutputHandler(
            tag="training", output_transform=lambda loss: {"batchloss": loss}, metric_names="all"
        ),
        event_name=Events.ITERATION_COMPLETED(every=100),
    )

    tb_logger.attach(
        train_evaluator,
        log_handler=OutputHandler(tag="training", metric_names=["Accuracy", "Loss"], another_engine=trainer),
        event_name=Events.EPOCH_COMPLETED,
    )

    tb_logger.attach(
        val_evaluator,
        log_handler=OutputHandler(tag="validation", metric_names=["Accuracy", "Loss"], another_engine=trainer),
        event_name=Events.EPOCH_COMPLETED,
    )

    trainer.run(train_loader, max_epochs=epochs)
    tb_logger.close()

if __name__ == '__main__':
    gin.parse_config_file('../config/mnist_config.gin')
    # Setup engine logger
    logger = logging.getLogger("ignite.engine.engine.Engine")
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    run()
