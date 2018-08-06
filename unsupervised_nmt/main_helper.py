import json
from collections import namedtuple

import numpy as np
import tensorflow as tf

from unmt_model import LanguageEncoderDecoder


TrainModel = namedtuple('TrainModel', ['model', 'output', 'loss', 'train_op'])


def load_vocabs(data_path):
    LANS = ['en', 'de']
    vocab_file_path = f'{data_path}/vocabs'

    with open(f'{vocab_file_path}/vocab_{LANS[0]}.json', 'r') as fp:
        vocab_en = json.load(fp)

    with open(f'{vocab_file_path}/vocab_{LANS[1]}.json', 'r') as fp:
        vocab_de = json.load(fp)
    return vocab_en, vocab_de


def get_models(hparams1, hparams2, g_hparams, sampling_probability, use_denoising=True, inputs=None, global_step=None):
    def get_train_ops(model_loss, hparams):
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            with tf.name_scope('Adam'):
                optimizer = tf.train.AdamOptimizer(learning_rate=hparams.learning_rate,
                                                   beta1=hparams.beta1,
                                                   beta2=hparams.beta2,
                                                   epsilon=hparams.epsilon)
                if hparams.gradient_clipping:
                    gvs = optimizer.compute_gradients(model_loss)
                    capped_gvs = map(lambda gv: gv if gv[0] is None else [tf.clip_by_value(gv[0], -2., 2.), gv[1]], gvs)
                    train_op = optimizer.apply_gradients(capped_gvs, global_step=global_step)
                else:
                    train_op = optimizer.minimize(model_loss, global_step=global_step)
        return train_op

    def wrap_models(lan1_enc_dec, lan2_enc_dec, sampling_probability, use_denoising=True, inputs=None):
        if inputs is None:
            _input_lan1 = tf.placeholder(tf.int32, shape=(hparams1.batch_size, hparams1.max_out_length))
            _input_lan2 = tf.placeholder(tf.int32, shape=(hparams2.batch_size, hparams2.max_out_length))

            _input_sequence_length1 = tf.placeholder(tf.int32, shape=[hparams1.batch_size])
            _input_sequence_length2 = tf.placeholder(tf.int32, shape=[hparams2.batch_size])
        else:
            _input_lan1 = inputs[0]
            _input_lan2 = inputs[1]

            _input_sequence_length1 = inputs[2]
            _input_sequence_length2 = inputs[3]

        output_den1, loss_den1 = lan1_enc_dec.denoising_model(_input_lan1, _input_sequence_length1,
                                                              sampling_probability)
        output_den2, loss_den2 = lan2_enc_dec.denoising_model(_input_lan2, _input_sequence_length2,
                                                              sampling_probability)

        output_back1, loss_back1 = lan1_enc_dec.backtranslation_model(_input_lan1, _input_sequence_length1,
                                                                      sampling_probability, lan2_enc_dec)
        output_back2, loss_back2 = lan2_enc_dec.backtranslation_model(_input_lan2, _input_sequence_length2,
                                                                      sampling_probability, lan1_enc_dec)
        train_op_den1 = get_train_ops(loss_den1, g_hparams)
        train_op_den2 = get_train_ops(loss_den2, g_hparams)
        train_op_back1 = get_train_ops(loss_back1, g_hparams)
        train_op_back2 = get_train_ops(loss_back2, g_hparams)

        train_model_den1 = TrainModel(model=lan1_enc_dec, output=output_den1, loss=loss_den1, train_op=train_op_den1)
        train_model_den2 = TrainModel(model=lan2_enc_dec, output=output_den2, loss=loss_den2, train_op=train_op_den2)
        train_model_back1 = TrainModel(model=lan1_enc_dec, output=output_back1, loss=loss_back1, train_op=train_op_back1)
        train_model_back2 = TrainModel(model=lan2_enc_dec, output=output_back2, loss=loss_back2, train_op=train_op_back2)

        if use_denoising:
            return train_model_den1, train_model_den2, train_model_back1, train_model_back2
        else:
            return train_model_back1, train_model_back2

    lan1_enc_dec = LanguageEncoderDecoder(hparams1)
    lan2_enc_dec = LanguageEncoderDecoder(hparams2, lan1_enc_dec)
    return wrap_models(lan1_enc_dec, lan2_enc_dec, sampling_probability,
                       use_denoising=use_denoising,
                       inputs=inputs)


def get_data(data_path, name, hparams):
    with tf.name_scope('inputs'):
        tfrecords_filename = f'{data_path}/{name}.tfrecords'
           
        if hparams.num_of_examples is None:
            num_of_examples = sum(1 for _ in tf.python_io.tf_record_iterator(tfrecords_filename))
            print(num_of_examples)
        else:
            num_of_examples = hparams.num_of_examples

        feature = {'sentence': tf.FixedLenFeature([], tf.string),
                   'length': tf.FixedLenFeature([], tf.int64)}

        filename_queue = tf.train.string_input_producer([tfrecords_filename], num_epochs=hparams.num_epochs)

        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)

        features = tf.parse_single_example(serialized_example, features=feature)

        sentence = tf.decode_raw(features['sentence'], tf.int32)
        sentence = tf.reshape(sentence, [hparams.max_out_length])

        s_length = tf.cast(features['length'], tf.int32)

        sentence_batch, s_length_batch = tf.train.shuffle_batch([sentence, s_length],
                                                                 batch_size=hparams.batch_size,
                                                                 capacity=2*hparams.batch_size,
                                                                 min_after_dequeue=hparams.batch_size,
                                                                 allow_smaller_final_batch=False)

        return sentence_batch, s_length_batch, num_of_examples


def ids2wordlist(idslist, vocabulary, idslen=None):
    sent = []
    if idslen is None:
        idslen = idslist.size
    print(idslist.shape)
    idslist = idslist.squeeze()
    vocabulary = {v: k for k, v in vocabulary.items()}
    
    for idx in range(idslen):
        ids = np.round(idslist[idx]).astype(np.int32)
        if ids in vocabulary:
            sent.append(vocabulary[ids])
    return sent