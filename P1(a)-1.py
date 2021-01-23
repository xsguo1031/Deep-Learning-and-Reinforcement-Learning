### code is from the lecture slides of EE 526X

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys



import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None

def main(_):
  # Import data
  #mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot = True)
  mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])
  w = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10]))
  y = tf.matmul(x, w) + b

  # Define loss and optimizer
  y_ = tf.placeholder(tf.int32, [None, 10])  
  
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = y))
  
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
  
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

'''
#Train
train_loops = 50000
for _ in range(train_loops):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict = {x:batch_xs, y_:batch_ys})
    
#Test:
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict = {x:mnist.test.images, y_:mnist.test.labels}))
'''


for train_loops in range(1000, 10000, 1000):
        for _ in range(train_loops):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            sess.run(train_step, feed_dict = {x:batch_xs, y_:batch_ys})
        #Test:
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print(f"Number of iteration is {train_loops}")
        print("Training Accuracy is:")
        print(sess.run(accuracy, feed_dict = {x:mnist.train.images, y_:mnist.train.labels}))
        print("Test Accuracy is:")
        print(sess.run(accuracy, feed_dict = {x:mnist.test.images, y_:mnist.test.labels}))
        




