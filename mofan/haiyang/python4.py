import tensorflow as tf

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = tf.multiply(input1, input2)

with tf.Session() as sess:
    print(sess.run(output, feed_dict={input1: [7, 9, 8, 10], input2: [4]}))
    print(sess.run(output, feed_dict={input1: [7, 9, 8, 10], input2: [1, 2, 3, 4]}))
