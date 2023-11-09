#MLP_Layer
import numpy as np

class MLP_layer():
    def __init__(self, activation, n_units, n_input):
        self.activation_function = activation
        self.n_units = n_units
        self.n_input = n_input
        weights = np.random.normal(0., 0.2, (n_units * n_input))
        self.weights = weights.reshape(n_units, n_input)
        self.bias = np.zeros((n_units, ))
    
    def forward(self, inputs):
        pre_activations = self.weights @ inputs + np.transpose(self.bias)
        return self.activation_function(pre_activations)
    