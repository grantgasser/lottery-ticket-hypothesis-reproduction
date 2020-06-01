import torch
import torch.nn.utils.prune as prune
import numpy as np

from copy import deepcopy



class PruneModel:
    def __init__(self, network):
        self.network = network
        self.original_network = deepcopy(network)
        self.pruning_rounds = 7  # set to maybe 5 later
        self.current_round = 1
        self.layers = []
        self.masks = {}
        self.p = 0.002  # p as in the paper
        self.pruning_rate = (self.p)**(1/self.pruning_rounds)
        print('pruning rate:', self.pruning_rate)
        self.percent_remaining_weights = 1.0

        # "Connections to outputs are pruned at half the rate of the rest of the network"
        self.p_output_layer = self.p/2.0
        self.pruning_rate_output_layer = (self.p_output_layer)**(1/self.pruning_rounds)
        print('pruning rate output layer:', self.pruning_rate_output_layer)
        self.percent_remaining_weights_output_layer = 1.0

    def prune(self):
        """
        Simple One-Shot Pruning for now

        TODO: Iterative pruning
        """
        for pruning_iteration in range(self.pruning_rounds):
            print('Pruning iteration:', pruning_iteration)

            # calculate remaining weights for this iteration
            self.percent_remaining_weights *= self.pruning_rate
            self.percent_remaining_weights_output_layer *= self.pruning_rate_output_layer
            print('Percent Remain:', self.percent_remaining_weights)
            print('Percent Remain:', self.percent_remaining_weights_output_layer)

            # calculate remaining weights for this pruning iteration
            for idx, (name, module) in enumerate(list(self.network.named_modules())):
                if isinstance(module, torch.nn.Linear):
                    self.layers.append(module)
                    # "Connections to outputs are pruned at half the rate of the rest of the network"
                    _, next_module = list(self.network.named_modules())[idx+1]
                    if isinstance(next_module, torch.nn.CrossEntropyLoss):
                        # prune lowest magnitude weights
                        prune.l1_unstructured(module, name='weight', amount=self.pruning_rate_output_layer)
                        self.masks[name] = module.weight_mask
                        #print(module.weight[:1, :10])
                        prune.remove(module, 'weight')
                        #print(name, list(module.named_parameters())[1])

                        # actual_percent_remaining = np.count_nonzero(module.weight.detach().numpy()) / (module.weight.shape[0] * module.weight.shape[1])
                        # print('Ensuring correct amount of weights have been pruned with module {}.  {:.2%}'.format(name, actual_percent_remaining))
                        #np.testing.assert_almost_equal(actual_percent_remaining, self.percent_remaining_weights_output_layer, decimal=3)

                    else:
                        # prune lowest magnitude weights
                        prune.l1_unstructured(module, name='weight', amount=self.pruning_rate)
                        self.masks[name] = module.weight_mask
                        #print(module.weight[:1, :10])
                        prune.remove(module, 'weight')

                       # print(name, list(module.named_parameters())[1])

                        # actual_percent_remaining = np.count_nonzero(module.weight.detach().numpy()) / (module.weight.shape[0] * module.weight.shape[1])
                        # print('Ensuring correct amount of weights have been pruned module: {}.  {:.2%}'.format(name, actual_percent_remaining))
                        #np.testing.assert_almost_equal(actual_percent_remaining, self.percent_remaining_weights, decimal=3)

                        # for hook in module._forward_pre_hooks.values():
                        #     if hook._tensor_name == 'weight':
                        #         for h in hook:
                        #             print(h)
                        #         break

            # for idx, (name, module) in enumerate(list(self.network.named_modules())):
            #     if isinstance(module, torch.nn.Linear):
            #         prune.remove(module, 'weight')
            #         print(name, list(module.named_parameters())[1][1])
            # # only do one iteration for now, essentially One-Shot pruning
            # break

            for l in self.layers:
                print()






