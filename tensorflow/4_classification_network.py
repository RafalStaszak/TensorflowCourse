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
batch_size = 256
num_batches = int(mnist.train.num_examples / batch_size)
input_height = 28
input_width = 28
n_classes = 10
display_step = 1

x = tf.placeholder(tf.float32, [None, 28 * 28])
y = tf.placeholder(tf.float32, [None, n_classes])

def network(x):
    layer_1 = tf.layers.dense(x, units=128, activation=tf.nn.relu)
    layer_1 = tf.layers.dense(x, units=512, activation=tf.nn.relu)
    layer_1 = tf.layers.dense(x, units=256, activation=tf.nn.relu)
    out = tf.layers.dense(x, units=10)
    return out


prediction = network(x)
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
        a[i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
        print(test_classes[i])
    plt.show()
