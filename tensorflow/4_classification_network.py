##################################################
##Import the required libraries and read the MNIST dataset
##################################################
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import time

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


learning_rate = 0.01
epochs = 20
batch_size = 64
num_batches = int(mnist.train.num_examples / batch_size)
input_height = 28
input_width = 28
n_classes = 10
display_step = 1

x = tf.placeholder(tf.float32, [None, 28 * 28])
y = tf.placeholder(tf.float32, [None, n_classes])

def network(x):
    layer_1 = tf.layers.dense(x, units=128, activation=tf.nn.relu)
    layer_2 = tf.layers.dense(layer_1, units=512, activation=tf.nn.relu)
    layer_3 = tf.layers.dense(layer_2, units=256, activation=tf.nn.relu)
    out = tf.layers.dense(layer_3, units=10)
    return out

def network_conv(x):
    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    layer_1 = tf.layers.conv2d(x, filters=32, kernel_size=3, strides=(1, 1), activation=tf.nn.relu)
    layer_2 = tf.layers.conv2d(layer_1, filters=64, kernel_size=3, strides=(1, 1), activation=tf.nn.relu)
    pool = tf.layers.max_pooling2d(layer_2, pool_size=[2, 2], strides=[2, 2])
    layer_3 = tf.layers.conv2d(pool, filters=32, kernel_size=3, strides=(1, 1), activation=tf.nn.relu)
    layer_flat = tf.layers.flatten(x)
    dense = tf.layers.dense(layer_flat, units=1000, activation=tf.nn.relu)
    out = tf.layers.dense(dense, units=10)
    return out


prediction = network_conv(x)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)
    for i in range(epochs):
        for j in range(num_batches):

            batch_x, batch_y = mnist.train.next_batch(batch_size)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y})
            if epochs % display_step == 0:
                print("Epoch:", '%04d' % (i + 1),
                      "cost=", "{:.9f}".format(loss),
                      "Training accuracy", "{:.5f}".format(acc))
    print('Optimization Completed')

    y1 = sess.run(prediction, feed_dict={x: mnist.test.images[:256]})
    test_classes = np.argmax(y1, 1)
    f, a = plt.subplots(1, 10, figsize=(10, 2))

    for i in range(10):
        a[i].imshow(np.reshape(mnist.test.images[i+10], (28, 28)))
        print(test_classes[i+10])
    plt.show()
