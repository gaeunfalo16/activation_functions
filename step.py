import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

x_data = [[0.1, 2.3, 4.5]]

X = tf.placeholder(tf.float32, shape=[None, 3])

W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.maximum(0.0, tf.sign(tf.matmul(X, W) + b))

with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())

    prediction = sess.run(hypothesis, feed_dict={X: x_data})
    print(prediction)