import os
import json
import math
from time import time

import tensorflow as tf
from nets import vgg_7
from tools.data_generator import DataGenerator

batch_size = 64
lr = 0.001

model_dir = 'checkpoints'
record_file = 'record.json'

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
    eval_x = tf.placeholder(tf.float32, [None, 75, 75, 2])
    eval_proba = tf.nn.sigmoid(vgg_7(x))

    session.run(tf.global_variables_initializer())

    # saver
    saver = tf.train.Saver()

    last_step = -1
    last_logloss = 0
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(session, ckpt.model_checkpoint_path)
        if os.path.exists(record_file):
            record_json = json.load(open(record_file, 'r'))
            last_acc = record_json['logloss']
            last_step = record_json['step']
        print 'Model restored from {}, last logloss: {}, last step: {}' \
            .format(ckpt.model_checkpoint_path, last_logloss, last_step)

    data_generator = DataGenerator();
    train_batch_gen = data_generator.train_generator()

    total_loss = 0
    count = 0
    for step in range(last_step+1, 100000000):
        t = time()
        batch_x, batch_y = train_batch_gen.next()
        d_cost = time() - t

        t = time()
        _, loss_out = session.run([train_step, loss], feed_dict={x: batch_x, y: batch_y})
        b_cost = time() - t

        count += 1
        if step % 20 == 0:
            avg_loss = total_loss / count
            print 'global step {}, avg loss {}, data time: {:.2f}, train time: {:.2f} s' \
                .format(step, avg_loss, d_cost, b_cost)
            total_loss = 0
            count = 0

        if step != 0 and step % 500 == 0:
            model_path = saver.save(session, os.path.join(model_dir, 'vgg_7'))
            if os.path.exists(record_file):
                j_dict = json.load(open(record_file))
            else:
                j_dict = {'accuracy': 0}

            j_dict['step'] = step
            json.dump(j_dict, open(record_file, 'w'), indent=4)
            print 'Save model at {}'.format(model_path)

        if step != 0 and step % 3000 == 0:
            print 'Evaluate validate set ... '
            val_batch_count, val_count = data_generator.get_validate_batch_count()
            val_batch_gen = data_generator.validate_generator()

            logloss = 0
            for i in xrange(val_batch_count):
                val_batch_x, val_batch_y = val_batch_gen.next()
                val_batch_proba = session.run(eval_proba, feed_dict={eval_x: val_batch_x})

                for index, proba in enumerate(val_batch_proba):
                    yi = val_batch_y[index]
                    logloss += yi * math.log(proba) + (1 - yi) * math.log(1 - proba)

            logloss /= val_batch_count

            print 'Log loss: {}'.format(logloss)

            # save model if get lower logloss
            if logloss > last_logloss:
                last_logloss = logloss
                model_path = saver.save(session, os.path.join(model_dir, 'best'))
                record_json = {'logloss': last_logloss, 'step': step}
                with open(record_file, 'w') as f:
                    json.dump(record_json, f, indent=4)

                print 'Get lower logloss, {}. Save model at {}, Save logloss at {}' \
                    .format(logloss, model_path, record_file)

def main():
    with tf.Session() as session:
        train(session)

if __name__ == '__main__':
    main()

