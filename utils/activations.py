import tensorflow as tf

def lrelu(x, leak=0.1, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def leaky_rectify(x, leakiness=0.2):
    assert leakiness <= 1
    ret = tf.maximum(x, leakiness * x)
    # import ipdb; ipdb.set_trace()
    return ret

def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale*tf.where(x>=0.0, x, alpha*tf.nn.elu(x))