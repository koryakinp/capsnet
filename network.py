import tensorflow as tf
from utils import *


def get_graph(X, y):
    conv1 = first_conv_layer(X)
    conv2 = second_conv_layer(conv1)

    caps1_raw = tf.reshape(conv2, [-1, 1152, 8], name="caps1_raw")
    caps1_output = squash(caps1_raw, name="caps1_output")

    batch_size = tf.shape(X)[0]
    W = get_weights(batch_size)

    temp1 = tf.expand_dims(caps1_output, -1, name="caps1_output_expanded")
    temp2 = tf.expand_dims(temp1, 2, name="caps1_output_tile")
    temp3 = tf.tile(temp2, [1, 1, 10, 1, 1], name="caps1_output_tiled")
    caps2_predicted = tf.matmul(W, temp3, name="caps2_predicted")

    weights = tf.zeros([batch_size, 1152, 10, 1, 1], name="route_weights")

    weights = routing_by_agreement(weights, caps2_predicted)
    weights = routing_by_agreement(weights, caps2_predicted)
    weights = routing_by_agreement(weights, caps2_predicted)

    weights = tf.nn.softmax(weights, dim=2, name="final_weights")
    prediction = tf.multiply(weights, caps2_predicted, name="prediction")
    weighted = tf.reduce_sum(prediction, axis=1, keep_dims=True, name="sum")
    caps2_output = squash(weighted, axis=-2, name="caps2_output_round_2")
    norm = safe_norm(caps2_output, axis=-2, keep_dims=True, name="norm")
    return tf.squeeze(norm, axis=[1, 3, 4], name="y_pred")


def get_weights(batch_size):
    W_init = tf.random_normal(shape=(1, 1152, 10, 16, 8), stddev=0.1)
    W = tf.Variable(W_init, name="W")
    return tf.tile(W, [batch_size, 1, 1, 1, 1], name="W_tiled")


def primary_capsules(input):
    with tf.variable_scope('primary_capsules'):
        capsules = tf.reshape(input, [1152, 8])

        W_init = tf.random_normal(
            shape=(1152, 10, 16, 8), stddev=0.1, name="W_init")
        W = tf.Variable(W_init, name="W")

        caps1_output = squash(capsules, 'primary_capsules_squash')

        caps1_output_expanded = tf.expand_dims(caps1_output, -1)
        caps1_output_tile = tf.expand_dims(caps1_output_expanded, 1)
        caps1_output_tiled = tf.tile(caps1_output_tile, [1, 10, 1, 1])

        return tf.matmul(W, caps1_output_tiled, name="caps2_predicted")


def first_conv_layer(input):
    with tf.variable_scope('first_conv_layer'):

        w = tf.truncated_normal([9, 9, 1, 256], stddev=0.1)
        b = tf.truncated_normal([256], stddev=0.1)

        weights = tf.Variable(w, name='weights')
        bias = tf.Variable(b, name='biases')

        conv = tf.nn.conv2d(
            input,
            weights,
            strides=[1, 1, 1, 1],
            padding='VALID') + tf.cast(bias, tf.float32)

        return tf.nn.relu(conv)


def second_conv_layer(input):
    with tf.variable_scope('second_conv_layer'):

        w = tf.truncated_normal([9, 9, 256, 256], stddev=0.1)

        weights = tf.Variable(w, name='weights')

        conv = tf.nn.conv2d(
            input,
            weights,
            strides=[1, 2, 2, 1],
            padding='VALID')

        return tf.nn.relu(conv)


def routing(input, weights, iteration):
    with tf.variable_scope('routing' + str(iteration)):
        temp2 = tf.nn.softmax(
            weights, dim=1, name="routing_weights")
        temp3 = tf.multiply(
            temp2, input, name="weighted_predictions")
        temp4 = tf.reduce_sum(
            temp3, axis=0, keep_dims=True, name="weighted_sum")
        return squash(temp4, axis=-2, name='routing_squash' + str(iteration))


def update_routing_weights(weights, input, prediction, iteration):
    with tf.variable_scope('routing_weight_update' + str(iteration)):
        tile = tf.tile(input, [1152, 1, 1, 1])
        agreement = tf.matmul(prediction, tile, transpose_a=True)
        return tf.add(weights, agreement)
