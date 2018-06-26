import os

import tensorflow as tf

from .utils import *

cur_path = os.path.dirname(__file__)
data_path = os.path.join(cur_path, '../Data')


def embedding_encoder(input, embedding_dim):
    return tf.contrib.layers.fully_connected(input, embedding_dim, activation=tf.nn.softmax, biases_initializer=None)


def encoder(input, hparams):
    encoder_cell = tf.nn.rnn_cell.GRUCell()