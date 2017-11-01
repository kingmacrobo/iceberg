import os
import json
import tensorflow as tf
from nets import vgg_7

batch_size = 64
lr = 0.001

model_dir = 'checkpoints'
acc_file = 'acc.json'

def train(session):

    # train net
    x = tf.placeholder(tf.float32, [batch_size, 75, 75, 2])
    y = tf.placeholder(tf.float32, [batch_size])

    logits = vgg_7(x)

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits))

    global_step = tf.Variable(0, name='global_step', trainable=False)

    train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(
        loss,
        global_step=global_step)

    # evaluate net
    tf.get_variable_scope().reuse_variables()
    eval_x = tf.placeholder(tf.float32, [batch_size, 75, 75, 2])
    eval_proba = tf.nn.sigmoid(vgg_7(x))

    session.run(tf.global_variables_initializer())

    # saver
    saver = tf.train.Saver()

    last_step = -1
    last_acc = 0
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(session, ckpt.model_checkpoint_path)
        if os.path.exists(acc_file):
            acc_json = json.load(open(acc_file, 'r'))
            last_acc = acc_json['accuracy']
            last_step = acc_json['step']
        print 'Model restored from {}, last accuracy: {}, last step: {}' \
            .format(ckpt.model_checkpoint_path, last_acc, last_step)




