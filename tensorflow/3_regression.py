import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt



x_data = np.random.rand(20).astype(np.float32)
y_data = x_data*1+3+np.random.uniform(0, 0.2, size=[20])

plt.plot(x_data, y_data, 'ro', label='Produced data')
plt.legend()
plt.show()

a=tf.Variable([0], dtype=tf.float32)
b=tf.Variable([0], dtype=tf.float32)

prediction = a*x_data+b


loss = tf.reduce_mean(tf.square((prediction-y_data)))
optimizer = tf.train.GradientDescentOptimizer(0.5)


train = optimizer.minimize(loss)
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in range(201):
    sess.run(train)

final_a = sess.run(a)
final_b = sess.run(b)
predicted_values = sess.run(prediction)

plt.plot(x_data, predicted_values, label='Predicted linear function')
plt.plot(x_data, y_data, 'ro', label='Original Data')
plt.legend()
plt.show()