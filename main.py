import numpy as np
import tensorflow as tf
from utils import *
from network import *

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

y = tf.placeholder(shape=[None, 10], dtype=tf.int64, name="y")
X = tf.placeholder(shape=[None, 28, 28, 1], dtype=tf.float32, name="X")

logits = get_graph(X, y)

y_pred = tf.argmax(logits, axis=1, name="y_proba")

loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)

correct = tf.equal(y, y_pred, name="correct")
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

optimizer = tf.train.AdamOptimizer()
training_op = optimizer.minimize(loss, name="training_op")

init = tf.global_variables_initializer()
saver = tf.train.Saver()

batch_size = 50
restore_checkpoint = False

iterations = int(mnist.train.num_examples/batch_size)
checkpoint_path = "./my_capsule_network"

with tf.Session() as sess:
    if restore_checkpoint and tf.train.checkpoint_exists(checkpoint_path):
        saver.restore(sess, checkpoint_path)
    else:
        init.run()

    for iteration in range(1, iterations + 1):
        X_batch, y_batch = mnist.train.next_batch(batch_size)
        # Run the training operation and measure the loss:
        _, loss_train = sess.run(
            [training_op, loss],
            feed_dict={X: X_batch.reshape([-1, 28, 28, 1]),
                       y: y_batch})

        print("\rIteration: {}/{} ({:.1f}%)  Loss: {:.5f}".format(
                  iteration, iterations,
                  iteration * 100 / iterations,
                  loss_train.mean()),
              end="")

    save_path = saver.save(sess, checkpoint_path)


n_iterations_test = mnist.test.num_examples // batch_size

with tf.Session() as sess:
    saver.restore(sess, checkpoint_path)

    loss_tests = []
    acc_tests = []
    for iteration in range(1, n_iterations_test + 1):
        X_batch, y_batch = mnist.test.next_batch(batch_size)
        loss_test, acc_test = sess.run(
                [loss, accuracy],
                feed_dict={X: X_batch.reshape([-1, 28, 28, 1]),
                           y: y_batch})
        loss_tests.append(loss_test)
        acc_tests.append(acc_test)
        print("\rEvaluating the model: {}/{} ({:.1f}%)".format(
                  iteration, n_iterations_test,
                  iteration * 100 / n_iterations_test),
              end=" " * 10)
    loss_test = np.mean(loss_tests)
    acc_test = np.mean(acc_tests)
    print("\rFinal test accuracy: {:.4f}%  Loss: {:.6f}".format(
        acc_test * 100, loss_test))
