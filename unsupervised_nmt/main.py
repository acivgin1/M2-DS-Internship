import os
import json
from collections import namedtuple

import numpy as np
import tensorflow as tf

from nn_helper import LanguageEncoderDecoder

TrainModel= namedtuple('TrainModel', ['output', 'loss', 'train_op'])

def get_hparams(vocab_en, vocab_de, batch_size):
    voc1_size = len(vocab_en) + 3
    voc2_size = len(vocab_de) + 3

    hparams1 = tf.contrib.training.HParams(
        embed_size=300,
        batch_size=batch_size,
        vocab_size=voc1_size,
        num_units=100,
        max_out_length=50,
        pad_token_id=np.array(voc1_size - 1, dtype=np.int32),
        end_token_id=np.array(voc1_size - 1, dtype=np.int32),
        sts_token_id=np.array(voc1_size - 2, dtype=np.int32),
        unk_token_id=np.array(voc1_size - 3, dtype=np.int32),
        name='en')

    hparams2 = tf.contrib.training.HParams(
        embed_size=hparams1.embed_size,
        batch_size=hparams1.batch_size,
        vocab_size=voc2_size,
        num_units=hparams1.num_units,
        max_out_length=hparams1.max_out_length,
        pad_token_id=np.array(voc2_size - 1, dtype=np.int32),
        end_token_id=np.array(voc2_size - 1, dtype=np.int32),
        sts_token_id=np.array(voc2_size - 2, dtype=np.int32),
        unk_token_id=np.array(voc2_size - 3, dtype=np.int32),
        name='de')
    return hparams1, hparams2


def load_vocabs(data_path):
    LANS = ['deen_de', 'deen_en']
    train_file_path = f'{data_path}/train_output'

    with open(f'{train_file_path}/{LANS[1]}_vocab.json', 'r') as fp:
        vocab_en = json.load(fp)

    with open(f'{train_file_path}/{LANS[0]}_vocab.json', 'r') as fp:
        vocab_de = json.load(fp)
    return vocab_en, vocab_de


def get_models(hparams1, hparams2, sampling_probability, use_denoising=True, inputs=None):
    def get_train_ops(model_loss, gradient_clipping=False):
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            with tf.name_scope('Adam'):
                optimizer = tf.train.AdamOptimizer()
                if gradient_clipping:
                    gvs = optimizer.compute_gradients(model_loss)
                    capped_gvs = map(lambda gv: gv if gv[0] is None else [tf.clip_by_value(gv[0], -10., 10.), gv[1]], gvs)
                    train_op = optimizer.apply_gradients(capped_gvs)
                else:
                    train_op = optimizer.minimize(model_loss)
        return train_op

    def wrap_models(lan1_enc_dec, lan2_enc_dec, sampling_probability, use_denoising=True, inputs=None):
        if inputs is None:
            _input_lan1 = tf.placeholder(tf.int32, shape=(hparams1.batch_size, hparams1.max_out_length))
            _input_lan2 = tf.placeholder(tf.int32, shape=(hparams2.batch_size, hparams2.max_out_length))

            _input_sequence_length1 = tf.placeholder(tf.int32, shape=[hparams1.batch_size])
            _input_sequence_length2 = tf.placeholder(tf.int32, shape=[hparams1.batch_size])
        else:
            _input_lan1 = inputs[0]
            _input_lan2 = inputs[1]

            _input_sequence_length1 = inputs[2]
            _input_sequence_length2 = inputs[3]

        output_den1, loss_den1 = lan1_enc_dec.denoising_model(_input_lan1, _input_sequence_length1, sampling_probability)
        output_den2, loss_den2 = lan2_enc_dec.denoising_model(_input_lan2, _input_sequence_length2, sampling_probability)

        output_back1, loss_back1 = lan1_enc_dec.backtranslation_model(_input_lan1, _input_sequence_length1,
                                                                      sampling_probability,
                                                                      lan2_enc_dec)
        output_back2, loss_back2 = lan2_enc_dec.backtranslation_model(_input_lan2, _input_sequence_length2,
                                                                      sampling_probability,
                                                                      lan1_enc_dec)
        train_op_den1 = get_train_ops(loss_den1)
        train_op_den2 = get_train_ops(loss_den2)
        train_op_back1 = get_train_ops(loss_back1)
        train_op_back2 = get_train_ops(loss_back2)

        train_model_den1 = TrainModel(output=output_den1, loss=loss_den1, train_op=train_op_den1)
        train_model_den2 = TrainModel(output=output_den2, loss=loss_den2, train_op=train_op_den2)
        train_model_back1 = TrainModel(output=output_back1, loss=loss_back1, train_op=train_op_back1)
        train_model_back2 = TrainModel(output=output_back2, loss=loss_back2, train_op=train_op_back2)

        if use_denoising:
            return train_model_den1, train_model_den2, train_model_back1, train_model_back2
        else:
            return train_model_back1, train_model_back2

    lan1_enc_dec = LanguageEncoderDecoder(hparams1)
    lan2_enc_dec = LanguageEncoderDecoder(hparams2, lan1_enc_dec)
    return wrap_models(lan1_enc_dec, lan2_enc_dec, sampling_probability, use_denoising=use_denoising, inputs=inputs)


def get_data(data_path, name, hparams):
    tfrecords_filename = f'{data_path}/{name}_sentences.tfrecords'

    num_of_examples = sum(1 for _ in tf.python_io.tf_record_iterator(tfrecords_filename))

    feature = {'sentence': tf.FixedLenFeature([], tf.string),
               'length': tf.FixedLenFeature([], tf.int64)}

    filename_queue = tf.train.string_input_producer([tfrecords_filename], num_epochs=1)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(serialized_example, features=feature)

    sentence = tf.decode_raw(features['sentence'], tf.int32)
    sentence = tf.reshape(sentence, [50])

    s_length = tf.cast(features['length'], tf.int32)

    sentence_batch, s_length_batch = tf.train.batch([sentence, s_length],
                                                    batch_size=hparams.batch_size,
                                                    capacity=2*hparams.batch_size,
                                                    allow_smaller_final_batch=True)

    return sentence_batch, s_length_batch, num_of_examples


def main(data_path, batch_size):
    vocab_en, vocab_de = load_vocabs(data_path)
    hparams1, hparams2 = get_hparams(vocab_en, vocab_de, batch_size)

    sentence1, sentence_length1, num1 = get_data(data_path, name='train_en', hparams=hparams1)
    sentence2, sentence_length2, num2 = get_data(data_path, name='train_de', hparams=hparams2)

    sampling_probability = tf.placeholder(tf.float32, shape=[])

    den1, den2, back1, back2 = get_models(hparams1, hparams2, sampling_probability,
                                          inputs=(sentence1,
                                                  sentence2,
                                                  sentence_length1,
                                                  sentence_length2))

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()

        all_vars = tf.trainable_variables()
        [tf.summary.histogram(var.name.split(':')[0], var) for var in all_vars]

        tf.summary.scalar('loss_den1', den1.loss)
        tf.summary.scalar('loss_den2', den2.loss)
        tf.summary.scalar('loss_back1', back1.loss)
        tf.summary.scalar('loss_back2', back2.loss)

        merged = tf.summary.merge_all()

        writer = tf.summary.FileWriter(f'{data_path}/output', sess.graph)
        saver = tf.train.Saver()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for epoch in range(25):
            # TODO: iterating will not work on the last batch
            for batch_num in range(num1 // hparams1.batch_size + 1):
                ress = tuple(sess.run([den1.loss, den2.loss, back1.loss, back2.loss,
                                       den1.train_op, den2.train_op, back1.train_op, back2.train_op,
                                       merged], feed_dict={sampling_probability: 0.7}))
                print(f'ep:{epoch}: {ress[0]:.4f}, {ress[1]:.4f}, {ress[2]:.4f}, {ress[3]:.4f}')

                writer.add_summary(ress[8])
                writer.add_session_log(tf.SessionLog(status=tf.SessionLog.START))

                save_path = saver.save(sess, f'{data_path}/model/nmt_model_epoch{epoch}.ckpt')
                print('Model saved in path: %s' % save_path)

        coord.request_stop()
        coord.join(threads)

        writer.close()

if __name__ == '__main__':
    cur_path = os.path.dirname(__file__)
    data_path = os.path.relpath('../Data', cur_path)
    main(data_path, batch_size=64)