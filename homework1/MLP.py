#MLP
import numpy as np
import MLP_layer 
import Util


class MLP():
    def __init__(self, layers):
        self.layer_list = layers
        self.mlp_list = []
        self.num_layers = len(layers)

        #for i in range(self.num_layers):
        #    self.mlp_list[i] = MLP_layer()




    """
    def backprop_step(self, x, target, epsilon):
        
        a = [x]
        for layer in self.layer_list[1:]:
            x = layer(x)
            a.append(x)
        y = x #prediction

        sigmoid_prime = a[-1] * (1 - a[-1])
        delta = 2*(y - target) * sigmoid_prime

        output_layer = self.layer_list[-1]
        output_layer.adapt(delta, a[-2], epsilon)

        for l in reversed(range(1, self.num_layers -1)):

            sigmoid_prime = a[l] * (1 - a[l])

            delta = (delta @ self.layer_list[l+1].weights.T) * sigmoid_prime

            self.layer_list[l].adapt(delta, a[l-1], epsilon)
        """