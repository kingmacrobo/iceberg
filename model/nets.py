import tensorflow as tf
from layers import conv2d, fully_connnected, flatten, maxpooling

def vgg_7(x, layers=3, base_channel=32, input_channel=2):
    output_channel = input_channel

    for i in range(layers):
        x = conv2d(x, [3, 3, input_channel, output_channel], 'conv_0_'+str(i))
        input_channel = output_channel

        x = conv2d(x, [3, 3, input_channel, output_channel], 'conv_1_'+str(i))
        input_channel = output_channel
        output_channel *= 2

        x = maxpooling(x)

    x = flatten(x)
    x = fully_connnected(x, 256, 'fc_1')
    x = fully_connnected(x, 1, 'fc_2', activation='no')

    logits = tf.squeeze(x)

    return logits