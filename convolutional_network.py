import numpy as np
import pandas as pd
import tensorflow as tf

def prep_image(data):
    data = np.array(data)
    data = (data - np.min(data)) / (np.max(data) - np.min(data)) ** 2
    return data.reshape(75, 75)[25:50, 25:50].reshape(-1)

def add_variable_summary(variable):
    with tf.name_scope('summaries'):
        tf.summary.scalar('{}_value'.format(variable.name), variable)

def accuracy_metric(pred_y, true_y):
    with tf.name_scope('accuracy'):
        # TODO AS: Simplify
        # acc = tf.reduce_mean(tf.to_float(tf.equal(tf.to_int32(pred_y), tf.to_int32(true_y))))
        # add_variable_summary(acc)
        acc = tf.metrics.accuracy(tf.to_int32(true_y), tf.to_int32(pred_y))
        return acc

def logloss_metric(pred_y, true_y):
    with tf.name_scope('loss'):
        # TODO AS: Simplify
        # eps = 1e-7
        # loss = -tf.reduce_mean(
        #     true_y * tf.log(pred_y + eps) + \
        #     (1 - true_y) * tf.log(1 - pred_y + eps),
        #     name='logloss')
        loss = tf.losses.log_loss(true_y, pred_y)
        add_variable_summary(loss)
        return loss

def convolutional_network(X, training):
    with tf.name_scope('convolutional_network'):
        conv1 = tf.layers.conv2d(
            inputs=X,
            filters=8,
            kernel_size=[5, 5],
            padding='same',
            activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
        print(pool1.get_shape())
        pool1_flat = tf.reshape(pool1, (-1, 12 * 12 * 8))
        dense = tf.layers.dense(inputs=pool1_flat, units=64, activation=tf.nn.elu)
        dropout = tf.layers.dropout(inputs=dense, rate=0.6, training=training)
        return tf.layers.dense(inputs=dropout, units=1, activation=tf.nn.sigmoid)

X = tf.placeholder(tf.float32, shape=(None, 25, 25, 1), name='X')
y = tf.placeholder(tf.float32, shape=(None, 1), name='y')

training = tf.placeholder_with_default(False, shape=(), name='training')
pred = convolutional_network(X, training)

accuracy = accuracy_metric(pred, y)
logloss = logloss_metric(pred, y)
training_op = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9).minimize(logloss)

train = pd.read_json('./data/train.json')
X_full = np.concatenate(train.band_2.apply(lambda data: prep_image(data)).values).reshape(-1, 25, 25, 1).astype('float32')
y_full = train.is_iceberg.values.astype('int32').reshape(-1, 1)
X_tr, y_tr, X_val, y_val = X_full[:1400], y_full[:1400], X_full[1400:], y_full[1400:]

batch_size = 128
n_batches = len(X_tr) // batch_size
n_epochs = 1000
summary = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter('./logdir', tf.get_default_graph())

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    for epoch in range(n_epochs):
        indicies = np.random.choice(len(X_tr), len(X_tr), replace=False)
        for batch in range(n_batches):
            first, last = batch * batch_size, (batch + 1) * batch_size
            X_batch, y_batch = X_tr[indicies[first:last]], y_tr[indicies[first:last]]
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

        accuracy_value, logloss_value, summary_value = sess.run([accuracy, logloss, summary], feed_dict={X: X_val, y: y_val, training: False})
        summary_writer.add_summary(summary_value, epoch)
        print('Epoch {}, acc {}, logloss {}'.format(epoch, accuracy_value, logloss_value))
