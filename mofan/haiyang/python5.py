import tensorflow as tf
import numpy as np
from datetime import datetime

def add_layer(inputs, in_size, out_size, activation_fuchtion = None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_fuchtion is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_fuchtion(Wx_plus_b)
    return outputs

x_data = np.linspace(-1, 1, 300, dtype = np.float32)[:, np.newaxis]
noise = np.random.normal(0, 0.01, x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise

xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

l1 = add_layer(xs, 1, 10, activation_fuchtion = tf.nn.relu)
prediction = add_layer(l1, 10, 1, activation_fuchtion = None)
loss = tf.reduce_mean(tf.reduce_mean(tf.square(ys - prediction), reduction_indices = [1]))
# can use any optimizer
train_step = tf.train.AdagradOptimizer(0.1).minimize(loss)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(100000):
    # training
    sess.run(train_step, feed_dict = {xs: x_data, ys: y_data})
    if i % 5000 == 0:
        # to see the step imporvement
        print(datetime.now())
        print(sess.run(loss, feed_dict = {xs: x_data, ys: y_data}))
