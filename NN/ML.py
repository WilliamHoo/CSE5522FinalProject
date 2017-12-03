# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

from SetData import SetData

FLAGS = None


def main(_):
  # Import data
  myTrainData = SetData("train-1.csv")

  x_train = myTrainData._Xtrain
  y_train = myTrainData._Ytrain
  x_test = myTrainData._Xtest
  y_test = myTrainData._Ytest

  # Create the model
  x = tf.placeholder(tf.float32, [None, 57])

  W1 = tf.Variable(tf.zeros([57, 2]))
  b1 = tf.Variable(tf.zeros([2]))
  y = tf.matmul(x, W1) + b1

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 2])

  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # outputs of 'y', and then average across the batch.
  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

  train_step = tf.train.AdamOptimizer(0.05).minimize(cross_entropy)

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  # Train
  for _ in range(100):
    sess.run(train_step, feed_dict={x: x_train, y_: y_train})
    print(sess.run(cross_entropy, feed_dict={x: x_train, y_: y_train}))
    #print(sess.run(W1, feed_dict={x: x_train, y_: y_train}))

  # Test trained model
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print("Below is the accuracy")
  print(sess.run(accuracy, feed_dict={x: x_test, y_: y_test}))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)