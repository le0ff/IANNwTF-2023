#Utils, including activation functions and more

#imports
import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import random

def data_processing(test_size=0.2):
    digits = load_digits()
    #create data tuple
    data = list(zip(digits.data, digits.target))

    #data arrays have not to be reshaped because they are already in shape (64,) unlike digits.images (8x8)

    #convert data to float32, scale values to range [0 to 1]
    prepared_data = [(data.astype(np.float32) / 16.0, target) for data, target in data]
    #onehot_matrix with np identity
    onehot_matrix = np.identity(10, dtype=np.float32)
    #replace target with onehot vectors
    prepared_data = [(data, onehot_matrix[target])for data, target in prepared_data]
    #shuffle
    np.random.shuffle(prepared_data)

    #splitting data in test und train data
    test_data = prepared_data[:int(len(prepared_data)*test_size)]
    train_data = prepared_data[int(len(prepared_data)*test_size):]
    #return train and test
    return train_data, test_data

def data_generator(minibatch_size, data):
    np.random.shuffle(data)
    X, y = zip(*data)
    for i in range(0, len(X), minibatch_size):
        yield np.array(X[i:i + minibatch_size]), np.array(y[i:i + minibatch_size])

#Have to apply the functions batch-wise, struggled for some time with this, but with some advice of cian we were able to solve the problem

#Sigmoid Activation Function
class Sigmoid:
    def __call__(self, inputs):
        self.y = np.array([1 / (1+np.exp(-input)) for input in inputs])
        return self.y
    
    #Sigmoid derivative
    def backwards(self, inputs):
        return np.array([y * (1. - y) * i] for y, i in zip(self.y, inputs))
    

#Softmax Activation Function
class Softmax:
    def __call__(self, inputs):
        #e_x = np.exp(inputs - np.max(inputs, axis=-1, keepdims=True))
        #return e_x / np.sum(e_x, axis=-1, keepdims=True)
        self.y = np.array([np.exp(i) / np.sum(np.exp(i)) for i in inputs])
        return self.y
    
    #Softmax derivative
    def backwards(self, inputs):
        return np.array([y * (inputs - (inputs * y).sum(axis=1)[:, np.newaxis]) for y in self.y])

#CCE-loss  
class CCE:
    def __call__(self, inputs, targets):
        #small constant to prevent warning
        epsilon = 1e-100 
        return np.array([-np.sum(t * np.log(i + epsilon)) for i, t in zip(inputs, targets)])

        # #"clip prediction" to avoid extreme values
        # inputs = np.clip(inputs, epsilon, 1 - epsilon)
        # #return -np.sum(targets * np.log(inputs), axis=-1, keepdims=True) / len(targets)
        # return -np.sum(targets * np.log(inputs)) / len(targets)
    
    #formula from tutorial, seems good right? 
    def backwards(self, activations, targets):
        #return (activations - targets) / len(targets)
        return np.array([(a - t)/len(t) for a, t in zip(activations, targets)])
    

#plot ten images with labels and predictions
def plot_images(images, labels_onehot, predictions):
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    fig.subplots_adjust(hspace=0.5)

    #decode one-hot-vectors to labels again
    labels = []
    for onehot in labels_onehot:
        for i in range(10):
            if onehot[i]:
                labels.append(i)

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(8, 8), cmap='gray')
        #take npargmax for highest probability in prediction vector
        ax.set_title(f"Label: {labels[i]}\nPredicted: {np.argmax(predictions[i])}")
        ax.axis('off')
    
    plt.show()

#plot loss curve
def plot_loss_curve(loss_history, window_size=50):
    smoothed_loss = np.convolve(loss_history, np.ones(window_size) / window_size, mode='valid')
    plt.plot(smoothed_loss, label='Loss over epochs')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()