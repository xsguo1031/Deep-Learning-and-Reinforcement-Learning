from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.utils import to_categorical, plot_model

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)


fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


input_shape=(train_images.shape[1], train_images.shape[1], 1)


batch_size = 128
kernel_size = 3
pool_size = 2
filters = 64
dropout = 0.2

train_images = train_images.reshape((-1, 28, 28, 1))
test_images = test_images.reshape((-1, 28, 28, 1))

train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# one-hot encode the training and testing labels
# train_labels = to_categorical(train_labels)
# test_labels = to_categorical(test_labels)




batch_size = 128
kernel_size = 3
pool_size = 2
filters = 64
dropout = 0.2


# model is a stack of CNN-ReLU-MaxPooling
model = Sequential()
model.add(Conv2D(filters=filters,
                 kernel_size=kernel_size,
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size))
model.add(Conv2D(filters=filters,
                 kernel_size=kernel_size,
                 activation='relu'))
model.add(MaxPooling2D(pool_size))
model.add(Conv2D(filters=filters,
                 kernel_size=kernel_size,
                 activation='relu'))
model.add(Flatten())
# dropout added as regularizer
model.add(Dropout(dropout))
# output layer is 10-dim one-hot vector
model.add(Dense(10))
model.add(Activation('softmax'))

# Take a look at the model summary
model.summary()




#input_shape=(train_images.shape[1], train_images.shape[1])
#model = keras.Sequential([
#    keras.layers.Flatten(input_shape=input_shape),
#    keras.layers.Dense(128, activation='relu'),
#    keras.layers.Dense(10, activation='softmax')
#])


#model.summary()
#


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


Train = model.fit(train_images, 
                  train_labels, 
                  validation_data=(test_images, test_labels),
                  batch_size=batch_size,
                  epochs=20)
plt.plot(Train.history['accuracy'], "r")
plt.plot(Train.history['val_accuracy'], "b")
plt.title('Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.legend(['train', "test"], loc='upper left')
plt.show()

# Test Accuracy
# Evaluate the model on test set
Test = model.evaluate(test_images, test_labels, batch_size =batch_size)
# Print test accuracy
print('\n', 'Test accuracy:', Test[1])




