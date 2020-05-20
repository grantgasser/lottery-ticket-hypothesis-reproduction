import torch

class BasePruning:
    """
    Base Pruning class. Used to prune the weights (remove some weights of a neural network by setting them to 0)
    """
    def __init__(self, model):
        self.model = model
        self.mask = {}
        self.init_weights = {}

    def store_init_weights(self):
        """
        TODO: handle different pruning on different layers (i.e. conv)
            - should be able to specify what type of layers to prune for each experiment
        """
        for name, param in self.model.named_parameters():
            if 'bias' not in name:
                print(type(param))
                self.init_weights[name] = param.data


class OneShotPruning(BasePruning):
    def __init__(self):
        super(OneShotPruning, self).__init__()

    def prune_layer(self, pruning_rate):
        pass


