import torch.nn as nn
import torch.nn.functional as f

from nonlinear_mor.utils.logger import getLogger


class BaseFullyConnectedNetwork(nn.Module):
    """Base class for fully-connected neural networks."""
    def __init__(self, layers_sizes):
        super(BaseFullyConnectedNetwork, self).__init__()

        if (layers_sizes is None or not len(layers_sizes) > 1
           or not all(size >= 1 for size in layers_sizes)):
            raise ValueError

        self.input_size = layers_sizes[0]
        self.number_of_layers = len(layers_sizes)
        self.layers_sizes = layers_sizes
        self.output_size = layers_sizes[-1]

        self.linear_layers = nn.ModuleList()
        self.linear_layers.extend([nn.Linear(int(self.layers_sizes[i]), int(self.layers_sizes[i+1]))
                                   for i in range(0, self.number_of_layers - 1)])

    def forward(self, x):
        pass

    def print_parameters(self):
        pass


class FullyConnectedNetwork(BaseFullyConnectedNetwork):
    """Class for neural networks with linear layers.

    This class represents neural networks that consist solely of linear layers
    and use the same activation function between each of these layers. The
    activation function is always applied after passing a linear layer, except
    for the last layer where no activation function is used.

    Parameters
    ----------
    layers_sizes
        List of numbers of neurons in the layers of the neural network.
        The first number is the input size, the last number the output size.
        The numbers in between determine the sizes of the hidden layers.
    activation_function
        Activation function to use between the linear layers.
    """

    def __init__(self, layers_sizes, activation_function=f.relu):
        super(FullyConnectedNetwork, self).__init__(layers_sizes)

        self.activation_function = activation_function

        self.logger = getLogger('nonlinear_mor.FullyConnectedNetwork')

    def get_init_params(self):
        return {'layers_sizes': self.layers_sizes, 'activation_function': self.activation_function}

    def forward(self, x):
        for i in range(0, self.number_of_layers - 2):
            x = self.activation_function(self.linear_layers[i](x))
        return self.linear_layers[self.number_of_layers-2](x)

    def print_parameters(self):
        self.logger.info("=> Parameters of neural network:")
        self.logger.info(f'Overall layers: {int(self.number_of_layers)}')
        self.logger.info(f'Overall neurons: {int(sum(self.layers_sizes))}')
        self.logger.info(f'Input neurons: {int(self.input_size)}')
        self.logger.info(f'Output neurons: {int(self.output_size)}')
        self.logger.info("Architecture:")
        self.logger.info(self)
