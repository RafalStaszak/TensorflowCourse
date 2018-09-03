import tensorflow as tf

elements = tf.constant([1, 2, 3, 4, 5])

add_all_elements = tf.reduce_sum(elements)
max_value_of_elements = tf.reduce_max(elements)
min_value_of_elements = tf.reduce_min(elements)


with tf.Session() as sess:
    total_sum, max_value, min_value = sess.run([add_all_elements, max_value_of_elements, min_value_of_elements])
    print('Total sum equals to {0}'.format(total_sum))
    print('Maximum equals to {0}'.format(max_value))
    print('Minimum equals to {0}'.format(min_value))