# import tensorflow
import tensorflow as tf
# make function to place nodes on devices
def variables_on_cpu(op):
    if op.type == "Variable":
        return "/cpu:0"
    else:
        return "/gpu:0"
# make nodes and place them with the above functon
with tf.device(variables_on_cpu):
    a = tf.Variable(3.0)
    b = tf.constant(4.0)
    c = a * b
# run session and print result
with tf.Session() as sess:
    sess.run(a.initializer)
    print(sess.run(c))
