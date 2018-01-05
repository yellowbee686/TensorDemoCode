import tensorflow as tf

x = [[[0., 0., 0.], [0., 0., 0.]], [[0., 0., 0.], [0., 0., 0.]]]
sess = tf.Session()
print(sess.run(tf.reduce_logsumexp(input_tensor=x, axis=-1)))
