import tensorflow as tf         # import tensorflow
with tf.device("/gpu:0"):       # place varialbe i simply on GPU 0
    i = tf.Variable(3)          # declare variable i with initial value 3 (integer)
tf.Session().run(i.initializer) # but tensorflow does not have a kernel for integers (=implemnetation of ...
                                # ... integers on a device, here GPUs) so this results in an error message
# this issue can be fixed by using "i = tf.Variable(3.0)" or "i = tf.Variable(3, dtype=tf.float32)
