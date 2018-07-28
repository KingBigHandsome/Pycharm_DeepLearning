import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# load hand written dataset
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

batch_size = 100

n_batch = mnist.train.num_examples // batch_size

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# define a simple neural network

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

predication = tf.nn.softmax(tf.matmul(x, W) + b)

# define loss function   using cross entropy
# loss = tf.reduce_mean(tf.square(y-predication))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=predication))

# define optimizer
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

# initialize variables
init = tf.global_variables_initializer()

# argmax return the position of the the max_value which equals to 1
correct_predication = tf.equal(tf.argmax(y, 1), tf.argmax(predication, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predication, tf.float32))

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(21):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})

        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})

        print('Iter ' + str(epoch) + ' Testing Accuracy ' + str(acc))