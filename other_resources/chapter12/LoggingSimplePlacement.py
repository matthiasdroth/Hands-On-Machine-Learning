import tensorflow as tf
with tf.device("/cpu:0"):   # place nodes on device (CPU)
    a = tf.Variable(3.0)
    b = tf.constant(4.0)
config = tf.ConfigProto()
config.log_device_placement = True
with tf.Session(config=config) as sess:  # run session
    sess.run(a.initializer) # initialize variable "a"
    print(sess.run(a * b))  # run calculation and print result
