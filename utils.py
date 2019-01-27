import numpy as np
import tensorflow as tf


def squash(s, axis=-1, name=None):
    with tf.name_scope(name, default_name="squash"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis, keep_dims=True)
        safe_norm = tf.sqrt(squared_norm + 1e-7)
        squash_factor = squared_norm / (1. + squared_norm)
        unit_vector = s / safe_norm
        return squash_factor * unit_vector


def safe_norm(s, axis=-1, keep_dims=False, name=None):
    with tf.name_scope(name, default_name="safe_norm"):
        norm = tf.reduce_sum(tf.square(s), axis=axis, keep_dims=keep_dims)
        return tf.sqrt(norm + 1e-7)


def routing_by_agreement(weights, input):
    temp1 = tf.nn.softmax(weights, dim=2)
    temp2 = tf.multiply(temp1, input)
    temp3 = tf.reduce_sum(temp2, axis=1, keep_dims=True)
    temp4 = squash(temp3, axis=-2)
    temp5 = tf.tile(temp4, [1, 1152, 1, 1, 1])
    temp6 = tf.matmul(input, temp5, transpose_a=True)
    return tf.add(weights, temp6)
