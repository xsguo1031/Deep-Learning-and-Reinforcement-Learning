from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.models import Sequential
from keras.layers import Dense, Activation, SimpleRNN, LSTM
from keras.utils import to_categorical, plot_model
from keras.datasets import mnist

import numpy as np
import random


# load mnist dataset

data = np.load('data.npy')
print(np.shape(data))

x=data[:,0:499]
y=data[:,1:500]

x_train=x[0:66,:]
y_train=y[0:66,:]
x_test=x[66:100,:]
y_test=y[66:100,:]


# compute the number of labels
num_labels = len(np.unique(y_train))

## convert to one-hot vector
#y_train = to_categorical(y_train)
#y_test = to_categorical(y_test)

# resize
x_train = x_train.reshape(-1, 499, 1)
x_test  = x_test.reshape(-1, 499, 1)
y_train = y_train.reshape(-1, 499, 1)
y_test = y_test.reshape(-1, 499, 1)


# network parameters
batch_size = 20
dropout = 0.2

# model is RNN with 256 units, input is 28-dim vector 28 timesteps
model = Sequential()
model.add(SimpleRNN(units=20,
                    dropout=dropout,
                    input_shape=(499,1),
                    return_sequences = True,
                    activation='relu'))
model.add(SimpleRNN(units=29,
                    dropout=dropout,
                    return_sequences = True,
                    activation='relu'))
model.add(Dense(1,activation = None))

model.summary()

# loss function for one-hot vector
# use of sgd optimizer
# accuracy is good metric for classification tasks
model.compile(loss='mse',
              optimizer='sgd',
              metrics=['mean_squared_error'])
# train the network
history = model.fit(x_train, y_train, epochs=20, batch_size=batch_size,validation_data=(x_test, y_test), verbose=2)

plt.plot(history.history['mean_squared_error'], "r")
plt.plot(history.history['val_loss'], "b")
plt.title('Model MSE')
plt.ylabel('MSE')
plt.xlabel('Epoch')
plt.legend(['Training', "Test"], loc='upper left')
plt.show()

# Test Accuracy
# Evaluate the model on test set
Test = model.evaluate(x_test, y_test, batch_size =batch_size)
# Print test accuracy
print('\n', 'Test accuracy:', Test[1])












