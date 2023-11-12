from MLP import MLP
from Util import data_generator, data_processing, plot_images, plot_loss_curve
import matplotlib.pyplot as plt
import numpy as np

#split in train and test data
test_size = 0.1
train_data, test_data = data_processing(test_size)

input_size = len(train_data[0][0])
output_size = len(train_data[0][1])

#layout for mlp
mlp_size = [input_size, 32, output_size]

#create model, choose values for epochs, learning_rate and batch_size
model = MLP(mlp_size)

n_epochs = 1000
learning_rate = 0.01
minibatch_size = 64

#create data_gen and list with respective batches
data_gen = data_generator(minibatch_size, train_data)
data = list(data_gen)

loss_history = np.zeros(n_epochs)

#(training is a generator here, training the model epoch-wise)
training = model.fit(data, learning_rate, n_epochs)

#start training, printing the loss and keep track of the loss_history
for (epoch, loss) in training:

    if (epoch + 1)% 100 == 0 or epoch in (0, n_epochs - 1):
        print(f"Epoch {epoch+1}/{n_epochs} Loss: {round(loss, 4)}")
    loss_history[epoch] = loss

#plot loss
plot_loss_curve(loss_history)

#test the trained model with test_data
test_gen = data_generator(minibatch_size, test_data)

#here, only one batch is tested
X_test, y_test = next(test_gen)
test_predictions = model.predict(X_test)

#plot function displays 10 images and their predictions and actual labels
plot_images(X_test[:10], y_test[:10], test_predictions[:10])