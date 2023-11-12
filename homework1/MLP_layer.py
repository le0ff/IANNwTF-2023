#MLP_Layer
import numpy as np

class MLP_layer:
    def __init__(self, activation_fn, n_units, n_input):
        #activation function (sigmoid or softmax in our case)
        self.activation_function = activation_fn
        
        self.n_units = n_units
        self.n_input = n_input
        #initialize weights with random normal distribution of shape number of inputs x number of units
        self.weights = np.random.normal(0., 0.2, (n_input, n_units))
        #initialize bias with zeros
        self.bias = np.zeros(n_units,)
    
    #forward pass
    def forward(self, inputs):
        self.input = inputs
        #matrixmultiplication with inputs @ weights, add bias
        pre_activations = np.dot(inputs, self.weights) + self.bias
        #apply activation function
        return self.activation_function(pre_activations)
    
    #backward pass
    def backward(self, gradient):
        #big thank you to cian for his hint to solve the matrixmultiplication like this
        #set weight gradient and bias gradient
        self.gradient_weights = (np.matmul(self.input[:, :, np.newaxis], gradient[:, np.newaxis, :])).mean(axis=0)
        #self.gradient_weights = (self.input @ gradient).mean(axis=0)
        self.gradient_bias = gradient.mean(axis=0)
        #return matrixmultiplication of gradient and weights (transposed)
        return np.dot(gradient, self.weights.T)