##################################################
##Import the required libraries and read the MNIST dataset
##################################################
import random

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import cv2
import os

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, reshape=False)

learning_rate = 0.001
epochs = 25
batch_size = 64
num_batches = int(mnist.train.num_examples / batch_size)
display_step = 34564743

x = tf.placeholder(tf.float32, [None, 28, 28, 1])
y = tf.placeholder(tf.float32, [None, 28, 28, 1])


def lrelu(x, alpha=0.1):
    return tf.maximum(alpha * x, x)


def working_encoder(x):
    conv1 = tf.layers.conv2d(x, filters=32, kernel_size=(3, 3), strides=(1, 1), padding='SAME', use_bias=True,
                             activation=lrelu, name='conv1')
    maxpool1 = tf.layers.max_pooling2d(conv1, pool_size=(2, 2), strides=(2, 2), name='pool1')
    conv2 = tf.layers.conv2d(maxpool1, filters=32, kernel_size=(3, 3), strides=(1, 1), padding='SAME',
                             use_bias=True, activation=lrelu, name='conv2')
    encoded = tf.layers.max_pooling2d(conv2, pool_size=(2, 2), strides=(2, 2), name='encoding')
    return encoded


def working_decoder(latent):
    conv3 = tf.layers.conv2d(latent, filters=32, kernel_size=3, strides=1, name='conv3', padding='SAME',
                             use_bias=True, activation=lrelu)
    upsample1 = tf.layers.conv2d_transpose(conv3, filters=32, kernel_size=3, padding='same', strides=2,
                                           name='upsample1')
    upsample2 = tf.layers.conv2d_transpose(upsample1, filters=32, kernel_size=3, padding='same', strides=2,
                                           name='upsample2')
    logits = tf.layers.conv2d(upsample2, filters=1, kernel_size=3, strides=1, name='logits', padding='SAME',
                              use_bias=True)
    decoded = tf.sigmoid(logits, name='recon')
    return logits, decoded


# def encoder(x):
#     encoder = tf.layers.conv2d(x, filters=8, kernel_size=3, activation=tf.nn.relu, padding='valid', strides=2)
#     encoder = tf.layers.conv2d(encoder, filters=16, kernel_size=3, activation=tf.nn.relu, padding='valid', strides=2)
#     encoder = tf.layers.conv2d(encoder, filters=16, kernel_size=3, activation=tf.nn.relu, padding='valid', strides=1)
#     encoder = tf.layers.conv2d(encoder, filters=32, kernel_size=3, activation=tf.nn.relu, padding='valid', strides=1)
#     #
#     lv = tf.layers.flatten(encoder)
#     return lv
#
#
# def decoder(latent):
#     decoder = tf.reshape(latent, (-1, 2, 2, 32))
#     decoder = tf.image.resize_images(decoder, (4, 4))
#     decoder = tf.layers.conv2d(decoder, filters=16, kernel_size=3, padding='same', activation=tf.nn.relu)
#     decoder = tf.image.resize_images(decoder, (6, 6))
#     decoder = tf.layers.conv2d(decoder, filters=16, kernel_size=3, padding='same', activation=tf.nn.relu)
#     decoder = tf.image.resize_images(decoder, (14, 14))
#     decoder = tf.layers.conv2d(decoder, filters=8, kernel_size=3, padding='same', activation=tf.nn.relu)
#     decoder = tf.image.resize_images(decoder, (28, 28))
#     decoder = tf.layers.conv2d(decoder, filters=1, kernel_size=3, padding='same')
#     return decoder


def load_images():
    images = os.listdir('images')
    data = [np.expand_dims(cv2.resize(cv2.imread(os.path.join('./images', image), cv2.IMREAD_GRAYSCALE), (28, 28)), -1)
            for image in images]
    return data


logits, output = working_decoder(working_encoder(x))

loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y)
cost = tf.reduce_mean(loss)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.global_variables_initializer()

overlay = 0.95

with tf.Session() as sess:
    sess.run(init)
    backgrounds = load_images()
    for i in range(epochs):
        for j in range(num_batches):
            batch_x, _ = mnist.train.next_batch(batch_size)
            background = np.array([random.choice(backgrounds)] * batch_size) / 255

            concat = background * overlay + batch_x * (1-overlay)
            sess.run(optimizer, feed_dict={x: concat, y: batch_x})

        batch_cost = sess.run(cost, feed_dict={x: concat, y: batch_x})
        print("Epoch:", '%04d' % (i + 1),
              "cost=", "{:.9f}".format(batch_cost))

        rnd_bckgs = np.stack([random.choice(backgrounds) for _ in range(20)]) / 255
        concat = mnist.test.images[:20] * (1-overlay) + rnd_bckgs * overlay
        output_images = sess.run(output, feed_dict={x: concat})

        idx = random.randint(0, 19)

        cv2.imshow('Output', output_images[idx])
        cv2.imshow('Concat', concat[idx])
        cv2.imshow('Original', mnist.test.images[:20][idx])
        cv2.waitKey(500)
