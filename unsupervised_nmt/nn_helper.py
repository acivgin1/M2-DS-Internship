# import os

import tensorflow as tf

# from .utils import *

# cur_path = os.path.dirname(__file__)
# data_path = os.path.join(cur_path, '../Data')


### lan1 - lan_mapper1                                                                 decoder1  (embeddings) - lan1
###                     embedder -- (embeddings) -- encoder -- (meaning) -- attention
### lan2 - lan_mapper2                                                                 decoder2  (embeddings) - lan2


def language_mapper(input, hparams):
    # we keep the same number of parameters
    bottleneck_dim = int(hparams.input_dim * hparams.embedding_dim / 2 / \
                         (hparams.input_dim + hparams.embedding_dim + 1) + 0.6)
    output = tf.contrib.layers.fully_connected(input, bottleneck_dim, activation=tf.nn.relu)  # , biases_initializer=None)
    return output


def embedder(input, hparams):
    output = tf.contrib.layers.fully_connected(input,
                                               hparams.embedding_dim,
                                               activation=tf.nn.softmax,
                                               biases_initializer=None)
    return output


def encoder(input, hparams):
    fw_cell = tf.nn.rnn_cell.GRUCell(num_units=hparams.num_units)
    bw_cell = tf.nn.rnn_cell.GRUCell(num_units=hparams.num_units)

    outputs, _, output_state_bw = tf.contrib.rnn.static_bidirectional_rnn(fw_cell, bw_cell, input, dtype=tf.float32)
    return outputs, output_state_bw


