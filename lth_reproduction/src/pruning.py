import torch
import torch.nn.utils.prune as prune
import numpy as np

from copy import deepcopy


class Model:
    def __init__(self, network):
        self.network = network
        self.original_network = deepcopy(network)
        self.pruning_rounds = 5  # set to maybe 5 later
        self.current_round = 1
        self.percent_remaining_weights = 1.0
        self.pruning_rate = 0.2
        self.masks = {}

        # "Connections to outputs are pruned at half the rate of the rest of the network"
        self.pruning_rate_output_layer = self.pruning_rate/2.0
        self.percent_remaining_weights_output_layer = 1.0

    def prune(self):
        """
        Simple One-Shot Pruning for now

        TODO: Iterative pruning and Random pruning (if random=T, the do prune.random_unstructured)
        """
        # self.percent_remaining_weights = ((self.pruning_rate) ** (1/self.pruning_rounds)) * self.percent_remaining_weights
        # p = 1 - self.percent_remaining_weights

        print('percent remaining: {}'.format(self.percent_remaining_weights))

        for pruning_iteration in range(self.pruning_rounds):
            # calculate remaining weights for this pruning iteration
            self.percent_remaining_weights = ((self.pruning_rate) ** (1 / self.pruning_rounds)) * self.percent_remaining_weights
            self.percent_remaining_weights_output_layer = ((self.pruning_rate_output_layer) ** (1 / self.pruning_rounds)) * self.percent_remaining_weights_output_layer

            for idx, (name, module) in enumerate(list(self.network.named_modules())):
                if isinstance(module, torch.nn.Linear):
                    # "Connections to outputs are pruned at half the rate of the rest of the network"
                    _, next_module = list(self.network.named_modules())[idx+1]
                    if isinstance(next_module, torch.nn.CrossEntropyLoss):
                        print('Pruning {} down to {:.2%} weights'.format(name, self.percent_remaining_weights_output_layer))
                        p = 1 - self.percent_remaining_weights_output_layer
                        prune.l1_unstructured(module, name='weight', amount=p)
                        self.masks[name] = module.weight_mask
                        prune.remove(module, 'weight')

                    else:
                        print('Pruning {} down to {:.2%} weights'.format(name, self.percent_remaining_weights))
                        p = 1 - self.percent_remaining_weights
                        prune.l1_unstructured(module, name='weight', amount=p)
                        self.masks[name] = module.weight_mask
                        prune.remove(module, 'weight')
                        print(module.weight[:2, :10])

                    print('% of weights remain in module {}: {}'.format(module, np.count_nonzero(module.weight.detach().numpy()) /
                        (module.weight.shape[0] * module.weight.shape[1])))

            # only do one iteration for now, essentially One-Shot pruning
            break





