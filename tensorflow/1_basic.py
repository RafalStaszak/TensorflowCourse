import tensorflow as tf


a = tf.constant(2)
b = tf.constant(3)

with tf.Session() as sess:
    print("a=2, b=3")
    print("Addition with constants: {0}".format(sess.run(a+b)))
    print("Multiplication with constants: {0}".format(sess.run(a*b)))


a = tf.placeholder(tf.int32)
b = tf.placeholder(tf.int32)

add = tf.add(a, b)
mul = tf.multiply(a, b)

# Launch the default graph.
with tf.Session() as sess:
    addition = sess.run(add, feed_dict={a: 2, b: 3})
    multiplication = sess.run(mul, feed_dict={a: 2, b: 3})
    print('Addition with variables: {0}'.format(addition))
    print('Multiplication with variables: {0}'.format(multiplication))

matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.],[2.]])


product = tf.matmul(matrix1, matrix2)
with tf.Session() as sess:
    result = sess.run(product)
    print('The result matrix multiplication equals {0}'.format(result))
