#MLP
import numpy as np
from MLP_layer import MLP_layer 
from Util import Sigmoid, Softmax, CCE

class MLP:
    def __init__(self, layers):
        self.layer_list = layers
        self.mlp_layers = []
        self.loss_fn = CCE()

        #create the layers, using sigmoid as activation (unless last layer)
        for i in range(len(self.layer_list)):
            if i == 0:
                activation_fn = Sigmoid()
                n_input = layers[i]
            # use softmax for the last layer
            elif i == len(self.layer_list) - 1:
                activation_fn = Softmax()
                n_input = layers[i - 1]
            else:
                activation_fn = Sigmoid()
                n_input = layers[i - 1]

            #create layers with number of units and n_inputs (number of units of current-1 layer)
            n_units = layers[i]
            layer = MLP_layer(activation_fn, n_units, n_input)
            self.mlp_layers.append(layer)

    #prediction, receives input, passes values through the layers and returns final activation
    def predict(self, inputs):
        activations = inputs
        # for every layer in layerlist, pass on the activation
        for layer in self.mlp_layers:
            activations = layer.forward(activations)

        return activations
    
    #backpropagation with prediction and targets as parameters
    def backpropagation(self, predictions, targets):
        #calculate gradient with derivative of cce-loss, then pass on gradient through the layer in reverse
        gradient = self.loss_fn.backwards(predictions, targets)
        for i in range(len(self.mlp_layers) -1, -1, -1):
            gradient = self.mlp_layers[i].backward(gradient)
    
    #fit (or train) with data, learning_rate n_epochs
    def fit(self, data, learning_rate=0.01, n_epochs=1000):
        #for for each epoch yield epoch and loss
        for epoch in range(n_epochs):
            total_loss = 0.0
            train_count = 0
            #for input, targets in minibatches
            for (X, y) in data:
                train_count += X.shape[0]

                predictions = self.predict(X)

                current_loss = self.loss_fn(predictions, y)

                total_loss += current_loss.sum()

                self.backpropagation(predictions, y)

                #update weights and biases based on gradient and learningrate
                for layer in self.mlp_layers:
                    layer.weights -= learning_rate * layer.gradient_weights
                    layer.bias -= learning_rate * layer.gradient_bias
            
            yield epoch, (total_loss/train_count)