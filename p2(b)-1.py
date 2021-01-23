import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.datasets import fashion_mnist
from keras import optimizers
from keras.layers import Dense

# load data
mnist = tf.keras.datasets.mnist
(x,y), (x_test, y_test) = mnist.load_data()
x = tf.keras.utils.normalize(x, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)
model = tf.keras.Sequential()

# Forward Propagation
model.add(tf.keras.layers.Flatten()) # To transform the 28*28 matrix become 784 dim vector
model.add(tf.keras.layers.Dense(units = 50,activation='relu', input_dim = 784)) #add one layer with 10  neurons
model.add(tf.keras.layers.Dense(units = 50, activation ='relu'))
model.add(tf.keras.layers.Dense(units = 10, activation ='softmax'))
#Backward Propagation 
#sgd = optimizers.SGD(lr=0.5)
model.compile(optimizer = 'sgd', loss = 'sparse_categorical_crossentropy', metrics = ["accuracy"])


Train = model.fit(x,y, epochs=50, batch_size = 500)
plt.plot(Train.history['acc'])
plt.title('Training Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.legend(['train'], loc='upper left')
plt.show()


test_loss, test_acc = model.evaluate(x_test, y_test, batch_size = 100)
print(test_loss, test_acc)