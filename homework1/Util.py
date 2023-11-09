#Utils, including activation functions and more

#imports
import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt


def data_generator(minibatch_size):
    #load digits from sklearn.datasets
    digits = load_digits()
    #get input (data of images) and target values 
    input, target = digits.data, digits.target
    #create data tuple
    data = list(zip(input, target))

    #data arrays have not be reshaped because they are already in shape (64,) unlike digits.images (8x8)

    #convert data to float32
    prepared_data = [(data.astype(np.float32), target) for data, target in data]
    #scale values to range [0 to 1]
    prepared_data = [(data / 16.0, target) for data, target in prepared_data]
    #onehot_matrix with np identity
    onehot_matrix = np.identity(10, dtype=np.float32)
    #replace target with onehot vectors
    prepared_data = [(data, onehot_matrix[target])for data, target in prepared_data]

    while True:
        #shuffle data
        np.random.shuffle(prepared_data)
        #for length of input data, stepsize of batchsize
        for i in range(0, len(prepared_data), minibatch_size):
            minibatch_data = []
            minibatch_targets = []

            #create minibatches with for loop, could also be done by slicing 
            for j in range(minibatch_size):
                if i + j < len(prepared_data):
                    data, target = prepared_data[i + j]
                    minibatch_data.append(data)
                    minibatch_targets.append(target)
            
            minibatch_data = np.array(minibatch_data, dtype=np.float32)
            minibatch_targets = np.array(minibatch_targets, dtype=np.float32)

            #ensures that minibatch is always same size, if not enough elements left it already starts over again
            if len(minibatch_data) == minibatch_size:
                yield minibatch_data, minibatch_targets

#Sigmoid Activation Function
class Sigmoid():
    def call(self, inputs):
        return 1 / (1+np.exp(-inputs))
    
    def backwards(self, inputs):
        y = Sigmoid.call(inputs)
        return y * (1 - y)
    

#Softmax Activation Function
class Softmax():
    def call(self, inputs):
        e_x = np.exp(inputs)
        return e_x / np.sum(e_x, axis=1, keepdims=True)


#CCE-loss  
class CCE():
    def loss(self, inputs, targets):
        return -np.sum(targets * np.log(inputs), axis=1)
    
    #formula from tutorial, seems good riiight? 
    def backwards(self, targets, activations):
        return activations - targets



##############
#TESTING-AREA#
##############

#plot function to check if digits correct
def plot_digit(input):
    image = input.reshape(8, 8)

    plt.figure()
    plt.imshow(image, cmap=plt.cm.gray)
    plt.show()


#plot-testing
gen = data_generator(64)
images, targets = next(gen)

# for i in range(len(images)):
#     plot_digit(images[i])
#     print(targets[i])