import torch
import torch.nn.utils.prune as prune
import numpy as np
from copy import deepcopy

import sys
sys.path.append('../src')
from train import train
from evaluate import test


class PruneModel:
    def __init__(self, network, batch_size, train_loader, val_loader, test_loader, optimizer, epochs, scheduler, device, pruning_rounds=1):
        """
        Class for pruning a model.

        Args:
            network (nn.Module): the network/model to be pruned
            pruning_rounds (int): the number of rounds in iterative pruning (1 if One Shot pruning)
        """
        self.network = network
        self.original_network = deepcopy(network)
        print('ORIGINAL NETWORK:')
        for name, param in self.original_network.named_parameters():
            print(name, param)
            break
        self.batch_size = batch_size
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.epochs = epochs
        self.scheduler = scheduler
        self.device = device
        self.pruning_rounds = pruning_rounds
        self.layers = []
        self.masks = {}
        self.p = 0.002  # p as in the paper
        self.pruning_rate = self.p ** (1/self.pruning_rounds)
        self.percent_remaining_weights_list = []

        # "Connections to outputs are pruned at half the rate of the rest of the network"
        # todo: not sure if this is done the same as the paper
        # should it be self.pruning_rate_output_layer = self.pruning_rate*2.0?
        self.p_output_layer = self.p*2.0
        self.pruning_rate_output_layer = self.p_output_layer ** (1/self.pruning_rounds)
        print(self.pruning_rate_output_layer)
        self.percent_remaining_weights_output_layer_list = []

        # predetermine % of weights at each pruning iteration
        for i in range(self.pruning_rounds+1):
            self.percent_remaining_weights_list.append(self.pruning_rate ** i)
            self.percent_remaining_weights_output_layer_list.append(self.pruning_rate_output_layer ** i)
        # print('\nRemaining weights: {}'.format(self.percent_remaining_weights_list))
        # print('Remaining weights output layer: {}'.format(self.percent_remaining_weights_output_layer_list))

    def prune(self):
        """
        Prune a network for pruning_rounds # of iterations. This function is the main driver of pruning, calling other
        functions such as _compute_masks, _apply_masks, and _retrain.
        """
        for pruning_iteration in range(self.pruning_rounds):
            print('-'*30)
            print('Pruning iteration:', pruning_iteration)
            print('-' * 30)
            print()
            # print('Percent Remain:', self.percent_remaining_weights_list[pruning_iteration])
            # print('Percent Remain:', self.percent_remaining_weights_output_layer_list[pruning_iteration])

            # compute masks
            self.masks = self._compute_masks()

            # reinit
            self.network = self._reinitialize(random=False)

            # apply the masks
            self._apply_masks()

            # verifying correct amount of parameters were pruned and correct amount is remaining
            self._test_pruning(pruning_iteration)

            # retrain after prune
            self._retrain()


    def _compute_masks(self):
        """
        Computes masks on self.network for a given iteration

        Returns:
            masks (Dict[str, torch.Tensor]: the masks for each layer
                a tensor of 0s and 1s having the same dimension as the parameter
        """
        masks = {}
        for idx, (name, param) in enumerate(self.network.named_parameters()):
            # todo: check linear, conv, etc. (isinstance())
            # todo: random sparse networks, so prune randomly, not based on magnitude
            if 'weight' in name:
                # get unpruned weights (nonzero)
                unpruned_weights = param[param != 0]

                # not output layer
                if idx < len(list(self.network.named_parameters())) - 1:
                    num_to_keep = int(self.pruning_rate * len(unpruned_weights))
                # output layer
                else:
                    num_to_keep = int(self.pruning_rate_output_layer * len(unpruned_weights))

                # find largest magnitude weights
                topk = torch.topk(torch.abs(param).view(-1), k=num_to_keep, largest=True)

                # create mask, keep largest magnitude weights by setting them to 1
                # remove smallest magnitude weights by setting them to 0
                mask = torch.zeros_like(param)
                mask.view(-1)[topk.indices] = 1

                masks[name] = mask

        return masks

    def _reinitialize(self, random=False):
        """
        Reinitialize the parameters. If random=True, reinitialize the parameters randomly
            Else: reinitialize parameters to original parameters (theta_0 in the paper)
        """
        if random:
            # create another instance of the neural network model class (randomly reinit)
            network_class = self.network.__class__
            new_random_network = network_class().to(self.device)
            return new_random_network
        else:
            # reinit to original weights
            return deepcopy(self.original_network)

    def _apply_masks(self):
        """
        Applies masks to self.network parameters.
            e.g. if this is a parameter [.1, -.2, .3, -.15, .05] and its mask is [0, 1, 0, 1, 1],
            the result is [0, -.2, 0, -.15, 0.5]
        """
        for name, param in self.network.named_parameters():
            if name in self.masks.keys():
                param.requires_grad_(requires_grad=False)
                param.mul_(self.masks[name])
                param.requires_grad_(requires_grad=True)
                # print(name)
                # print(param)
                # print(self.masks[name])
                # print()

    def _test_pruning(self, pruning_iteration):
        """
        Verify correct amount of weights have been pruned
        """
        for idx, (name, param) in enumerate(self.network.named_parameters()):
            if name in self.masks.keys():
                # not output layer
                if idx < len(list(self.network.named_parameters())) - 1:
                    theoretical_unpruned = int(
                        (self.pruning_rate ** (pruning_iteration + 1) * len(param.view(-1)))
                    )
                # output layer
                else:
                    theoretical_unpruned = int(
                        (self.pruning_rate_output_layer ** (pruning_iteration + 1) * len(param.view(-1)))
                    )

                actual_unpruned_param = len(param[param != 0])
                actual_nonzero_mask = torch.sum(self.masks[name])

                # all these should tell us how many weights/params still remain at a given pruning iteration
                diff = (theoretical_unpruned - actual_unpruned_param)
                assert (abs(diff) < 3)
                diff2 = (actual_unpruned_param - actual_nonzero_mask)
                assert (abs(diff2) < 3)

    def _retrain(self):
        """
        Retrains the network after pruning and weight reinitialization.
        """
        # run the training loop
        for epoch in range(1, self.epochs + 1):
            stop, stopping_iteration = train(
                self.network, self.device, self.train_loader, self.val_loader, self.test_loader, self.optimizer, epoch
            )

            self.scheduler.step()

            # test after each epoch
            test(self.network, self.device, self.test_loader)

            if stop:
                print('Stopped at overall iteration {}\n'.format(
                    stopping_iteration + ((len(self.train_loader.dataset) / self.batch_size) * (epoch - 1))))
                break

        # if save_model:
        #     torch.save(model.state_dict(), model.__class__.__name__ + '_' + dataset + ".pt")

