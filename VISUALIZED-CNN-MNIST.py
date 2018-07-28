"""
A simple MNIST classifier which displays summaries in TensorBoard.
This is an unimpressive MNIST model, but it is a good example of using
tf.name_scope to make a graph legible in the TensorBoard graph explorer, and of
naming summary tags so that they are grouped meaningfully in TensorBoard.
It demonstrates the functionality of every TensorBoard dashboard.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.tensorboard.plugins import projector

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

max_steps = 1001

image_num = 3000

DIR = "/home/dart/PycharmProjects/test/"


sess = tf.Session()

# Import data
embedding = tf.Variable(tf.stack(mnist.test.images[:image_num]), trainable=False, name='embedding')


#
def variable_summaries(var):
    """
    Attach a lot of summaries to a Tensor (for TensorBoard visualization).
    """
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


# Input placeholders
with tf.name_scope('Input'):
    x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
    y = tf.placeholder(tf.float32, shape=[None, 10], name='y')
    with tf.name_scope('x_image'):
        x_image = tf.reshape(x, [-1, 28, 28, 1], name='x_image')
        tf.summary.image('input',x_image,10)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


'''
Adding the first CNN layer. It consists of a Convolution and a Pooling
The meaning of the Weight tensor shape is as follows:
    the first two numbers defines the size of the filter;
    the third number means the input channel size of the input image. gray-scale image:1, RGB iamge:3
    the last parameter represents the number of the filter
The meaning of the bias tensor shape is the number of output channels

The shape of output image is [14,14,32]
'''
with tf.name_scope('Conv1'):
    with tf.name_scope('w_Conv1'):
        W_conv1 = weight_variable([5, 5, 1, 32])
    with tf.name_scope('b_Conv1'):
        b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

"""
Adding the second layer
The shape of output image is [7,7,64]
"""
with tf.name_scope('Conv2'):
    with tf.name_scope('w_Conv2'):
        W_conv2 = weight_variable([5, 5, 32, 64])
    with tf.name_scope('b_Conv2'):
        b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

"""
Adding the first fully-connected layer
"""
with tf.name_scope('fc1'):
    with tf.name_scope('w_fc1'):
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
    with tf.name_scope('b_fc1'):
        b_fc1 = bias_variable([1024])
    # reshape the flat image to a vector
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

"""
Finally, Adding the softmax layer
"""
with tf.name_scope('output'):
    with tf.name_scope('w_output'):
        W_fc2 = weight_variable([1024, 10])
    with tf.name_scope('b_output'):
        b_fc2 = bias_variable([10])
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


# cross_entropy = -tf.reduce_sum(y * tf.log(y_conv))
# train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


lr = tf.Variable(0.0001, dtype=tf.float32)
new_lr = tf.multiply(lr, 0.95)
update = tf.assign(lr, new_lr)

"""
Train
"""
with tf.name_scope('train'):
    # Loss function: Cross Entropy
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_conv))
    tf.summary.scalar('loss', loss)
    train_step = tf.train.AdamOptimizer(lr).minimize(loss)
    # Calculate the accuracy
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    tf.summary.scalar('accuracy', accuracy)

"""
Merge all of summaries
"""
merged = tf.summary.merge_all()

#
batch_size = 100
#
n_batch = 5000

sess = tf.Session()
sess.run(tf.global_variables_initializer())

"""
Generate the metadata file

"""
# if tf.gfile.Exists(DIR + 'projector/projector/metadata.tsv'):
#     tf.gfile.DeleteRecursively(DIR + 'projector/projector/metadata.tsv')

with open(DIR + 'projector/projector/metadata.tsv', 'w') as f:
    labels = sess.run(tf.argmax(mnist.test.labels[:], 1))
    for i in range(image_num):
        f.write(str(labels[i]) + '\n')

projector_writer = tf.summary.FileWriter(DIR + 'projector/projector', sess.graph)
saver = tf.train.Saver()
config = projector.ProjectorConfig()
embed = config.embeddings.add()
embed.tensor_name = embedding.name
embed.metadata_path = DIR + 'projector/projector/metadata.tsv'
embed.sprite.image_path = DIR + 'projector/data/mnist_10k_sprite.png'
embed.sprite.single_image_dim.extend([28, 28])
projector.visualize_embeddings(projector_writer, config)

train_writer = tf.summary.FileWriter('logs/train', sess.graph)
test_writer = tf.summary.FileWriter('logs/test', sess.graph)

for i in range(5000):

    batch = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch[0], y: batch[1], keep_prob: 0.7})

    if (i + 1) % 100 == 0:
        sess.run(tf.assign(lr, tf.multiply(lr, 0.95)))
        print(sess.run(lr))
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        summary, _ = sess.run([merged, train_step], feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0},
                              options=run_options,
                              run_metadata=run_metadata)
        projector_writer.add_run_metadata(run_metadata, 'step%03d' % i)
        projector_writer.add_summary(summary, i)

        #
        summary = sess.run(merged, feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})
        train_writer.add_summary(summary, i)

        #
        batch_xs, batch_ys = mnist.test.next_batch(batch_size)
        summary = sess.run(merged, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})
        test_writer.add_summary(summary, i)

        train_accuracy = sess.run(accuracy, feed_dict={
            x: batch[0], y: batch[1], keep_prob: 1.0})

        print("step %d, training accuracy %6f" % (i + 1, train_accuracy))
        print("test accuracy %6f" % sess.run(accuracy, feed_dict={
            x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0}))

saver.save(sess, DIR + 'projector/projector/a_model.ckpt', global_step=max_steps)
projector_writer.close()