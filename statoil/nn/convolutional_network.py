import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime

def prep_image(data):
    data = np.array(data)
    data = (data - np.min(data)) / (np.max(data) - np.min(data)) ** 2
    return data.reshape(75, 75).reshape(-1)

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
            filters=64,
            kernel_size=(3, 3),
            padding='valid',
            strides=(1, 1),
            activation=tf.nn.relu)

        pool1 = tf.layers.max_pooling2d(
            inputs=conv1,
            pool_size=(2, 2),
            strides=(2, 2))

        dropout1 = tf.layers.dropout(
            inputs=pool1,
            rate=0.2,
            training=training)

        conv2 = tf.layers.conv2d(
            inputs=dropout1,
            filters=128,
            kernel_size=(3, 3),
            padding='valid',
            strides=(1, 1),
            activation=tf.nn.relu)

        pool2 = tf.layers.max_pooling2d(
            inputs=conv2,
            pool_size=(2, 2),
            strides=(2, 2))

        dropout2 = tf.layers.dropout(
            inputs=pool2,
            rate=0.2,
            training=training)

        conv3 = tf.layers.conv2d(
            inputs=dropout2,
            filters=64,
            kernel_size=(3, 3),
            padding='valid',
            strides=(1, 1),
            activation=tf.nn.relu)

        pool3 = tf.layers.max_pooling2d(
            inputs=conv3,
            pool_size=(2, 2),
            strides=(2, 2))

        dropout3 = tf.layers.dropout(
            inputs=pool3,
            rate=0.2,
            training=training)

        conv4 = tf.layers.conv2d(
            inputs=dropout3,
            filters=64,
            kernel_size=(3, 3),
            padding='valid',
            strides=(1, 1),
            activation=tf.nn.relu)

        pool4 = tf.layers.max_pooling2d(
            inputs=conv4,
            pool_size=(2, 2),
            strides=(2, 2))

        dropout4 = tf.layers.dropout(
            inputs=pool4,
            rate=0.2,
            training=training)

        print(dropout4.get_shape())
        dropout4_flat = tf.reshape(dropout4, (-1, 2 * 2 * 64))

        dense1 = tf.layers.dense(
            inputs=dropout4_flat,
            units=512,
            activation=tf.nn.elu)

        dropout5 = tf.layers.dropout(
            inputs=dense1,
            rate=0.2,
            training=training)

        dense2 = tf.layers.dense(
            inputs=dropout5,
            units=256,
            activation=tf.nn.elu)

        dropout6 = tf.layers.dropout(
            inputs=dense2,
            rate=0.2,
            training=training)

        dense3 = tf.layers.dense(
            inputs=dropout6,
            units=1,
            activation=tf.nn.sigmoid)

        return dense3

X = tf.placeholder(tf.float32, shape=(None, 75, 75, 1), name='X')
y = tf.placeholder(tf.float32, shape=(None, 1), name='y')

training = tf.placeholder_with_default(False, shape=(), name='training')
pred = convolutional_network(X, training)

accuracy = accuracy_metric(pred, y)
logloss = logloss_metric(pred, y)
# training_op = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9).minimize(logloss)
training_op = tf.train.AdamOptimizer().minimize(logloss)
compute_gradients = tf.gradients(logloss, tf.trainable_variables())

train = pd.read_json('./data/train.json')
X_full = np.concatenate(train.band_2.apply(lambda data: prep_image(data)).values).reshape(-1, 75, 75, 1).astype('float32')
y_full = train.is_iceberg.values.astype('int32').reshape(-1, 1)
X_tr, y_tr, X_val, y_val = X_full[:1400], y_full[:1400], X_full[1400:], y_full[1400:]

batch_size = 32
n_batches = len(X_tr) // batch_size
n_epochs = 1000
summary = tf.summary.merge_all()
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
summary_writer = tf.summary.FileWriter('./logdir/run-{}'.format(now), tf.get_default_graph())

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    for epoch in range(n_epochs):
        indicies = np.random.choice(len(X_tr), len(X_tr), replace=False)
        for batch in range(n_batches):
            first, last = batch * batch_size, (batch + 1) * batch_size
            X_batch, y_batch = X_tr[indicies[first:last]], y_tr[indicies[first:last]]
            grads, _ = sess.run([compute_gradients, training_op], feed_dict={X: X_batch, y: y_batch})

        accuracy_value, logloss_value, summary_value = sess.run([accuracy, logloss, summary], feed_dict={X: X_val, y: y_val, training: False})
        summary_writer.add_summary(summary_value, epoch * n_batches + batch)
        print('Epoch {}, acc {}, logloss {}'.format(epoch, accuracy_value, logloss_value))
