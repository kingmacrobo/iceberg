import tensorflow as tf
import numpy as np

def weight_variable(shape):
    return tf.get_variable("weights",
                           shape,
                           initializer=tf.truncated_normal_initializer(stddev=0.1))

def bias_variable(shape):
    return tf.get_variable("biases", shape, initializer=tf.constant_initializer(0.1))

# weight shape: [filter_h, filter_w, input_channels, output_channels]
def conv2d(x, w_shape, scope, activation='relu'):
    with tf.variable_scope(scope):
        w = weight_variable(w_shape)
        b = bias_variable([w_shape[-1]])
        x = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)

        '''
        #batch normalization
        if bn:
            fc_mean, fc_var = tf.nn.moments(
                x,
                axes=[0, 1, 2]
            )
            out_size = w_shape[-1]
            scale = tf.Variable(tf.ones([out_size]))
            shift = tf.Variable(tf.zeros([out_size]))
            epsilon = 0.00001

            # apply moving average for mean and var when train on batch
            ema = tf.train.ExponentialMovingAverage(decay=0.5)

            def mean_var_with_update():
                ema_apply_op = ema.apply([fc_mean, fc_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(fc_mean), tf.identity(fc_var)

            mean, var = mean_var_with_update()
            x = tf.nn.batch_normalization(x, mean, var, shift, scale, epsilon)
        '''

        if activation == 'sigmoid':
            return tf.nn.sigmoid(x)
        elif activation == 'no':
            return x

        return tf.nn.relu(x)

def fully_connnected(x, out_channels, scope, activation='relu'):
    with tf.variable_scope(scope):
        shape = x.get_shape().as_list()
        w_shape = [shape[-1], out_channels]
        w = weight_variable(w_shape)
        b = bias_variable(out_channels)
        x = tf.matmul(x, w) + b

        if activation == 'sigmoid':
            return tf.nn.sigmoid(x)

        return tf.nn.relu(x)

def flatten(x):
    shape = x.get_shape().as_list()
    dim = np.prod(shape[1:])
    return tf.reshape(x, [-1, dim])

def maxpooling(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')
