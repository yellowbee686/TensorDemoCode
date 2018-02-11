import tensorflow as tf

# x = [[[0., 0., 0.], [0., 0., 0.]], [[0., 0., 0.], [0., 0., 0.]]]
# sess = tf.Session()
# print(sess.run(tf.reduce_logsumexp(input_tensor=x, axis=-1)))

# a = tf.constant([2, 3], name='a')
# b = tf.constant([[0, 1], [2, 3], [4, 5]], name='b')
# x = tf.multiply(a, b, name='mul')
# with tf.Session() as sess:
#     print(sess.run(x))
#
# a = tf.constant([2, 2], name='a')
# b = tf.constant([[0, 1], [2, 3]], name='b')
# with tf.Session() as sess:
# 	print(sess.run(tf.div(b, a)))             # [[0 0] [1 1]]
# 	print(sess.run(tf.divide(b, a)))          # [[0. 0.5] [1. 1.5]]
# 	print(sess.run(tf.truediv(b, a)))         # [[0. 0.5] [1. 1.5]]
# 	print(sess.run(tf.floordiv(b, a)))        # [[0 0] [1 1]]
# 	# print(sess.run(tf.realdiv(b, a)))         # # Error: only works for real values
# 	print(sess.run(tf.truncatediv(b, a)))     # [[0 0] [1 1]]
# 	print(sess.run(tf.floor_div(b, a)))       # [[0 0] [1 1]]

W = tf.Variable(2)
U = tf.Variable(2 * W.initialized_value())
