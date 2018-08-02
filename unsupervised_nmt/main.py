import os
import sys
import json
from collections import namedtuple

import numpy as np
import tensorflow as tf

from nn_helper import LanguageEncoderDecoder


TrainModel= namedtuple('TrainModel', ['output', 'loss', 'train_op'])

def get_hparams(vocab_en, vocab_de, batch_size, num_epochs):
    voc1_size = len(vocab_en) + 3
    voc2_size = len(vocab_de) + 3

    hparams1 = tf.contrib.training.HParams(
        embed_size=300,
        batch_size=batch_size,
        vocab_size=voc1_size,
        num_units=300,
        max_out_length=50,
        pad_token_id=np.array(voc1_size - 1, dtype=np.int32),
        end_token_id=np.array(voc1_size - 1, dtype=np.int32),
        sts_token_id=np.array(voc1_size - 2, dtype=np.int32),
        unk_token_id=np.array(voc1_size - 3, dtype=np.int32),
        name='en',
        num_epochs=num_epochs,
        num_of_examples=4069955)

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
        name='de',
        num_epochs=num_epochs,
        num_of_examples=4069955)

    return hparams1, hparams2


def load_vocabs(data_path):
    LANS = ['deen_en', 'deen_de']
    train_file_path = f'{data_path}/train_output'

    with open(f'{train_file_path}/{LANS[0]}_vocab.json', 'r') as fp:
        vocab_en = json.load(fp)

    with open(f'{train_file_path}/{LANS[1]}_vocab.json', 'r') as fp:
        vocab_de = json.load(fp)
    return vocab_en, vocab_de


def get_models(hparams1, hparams2, sampling_probability, use_denoising=True, inputs=None, global_step=None):
    def get_train_ops(model_loss, gradient_clipping=False):
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            with tf.name_scope('Adam'):
                optimizer = tf.train.AdamOptimizer()
                if gradient_clipping:
                    gvs = optimizer.compute_gradients(model_loss)
                    capped_gvs = map(lambda gv: gv if gv[0] is None else [tf.clip_by_value(gv[0], -10., 10.), gv[1]], gvs)
                    train_op = optimizer.apply_gradients(capped_gvs, global_step=global_step)
                else:
                    train_op = optimizer.minimize(model_loss, global_step=global_step)
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

        output_den1, loss_den1 = lan1_enc_dec.denoising_model(_input_lan1, _input_sequence_length1,
                                                              sampling_probability)
        output_den2, loss_den2 = lan2_enc_dec.denoising_model(_input_lan2, _input_sequence_length2,
                                                              sampling_probability)

        output_back1, loss_back1 = lan1_enc_dec.backtranslation_model(_input_lan1, _input_sequence_length1,
                                                                      sampling_probability, lan2_enc_dec)
        output_back2, loss_back2 = lan2_enc_dec.backtranslation_model(_input_lan2, _input_sequence_length2,
                                                                      sampling_probability, lan1_enc_dec)
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
    return wrap_models(lan1_enc_dec, lan2_enc_dec, sampling_probability,
                       use_denoising=use_denoising,
                       inputs=inputs)


def get_data(data_path, name, hparams):
    with tf.name_scope('inputs'):
        tfrecords_filename = f'{data_path}/{name}_sentences.tfrecords'

        if hparams.num_of_examples is None:
            num_of_examples = sum(1 for _ in tf.python_io.tf_record_iterator(tfrecords_filename))
        else:
            num_of_examples = hparams.num_of_examples

        feature = {'sentence': tf.FixedLenFeature([], tf.string),
                   'length': tf.FixedLenFeature([], tf.int64)}

        filename_queue = tf.train.string_input_producer([tfrecords_filename], num_epochs=hparams.num_epochs)

        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)

        features = tf.parse_single_example(serialized_example, features=feature)

        sentence = tf.decode_raw(features['sentence'], tf.int32)
        sentence = tf.reshape(sentence, [50])

        s_length = tf.cast(features['length'], tf.int32)

        sentence_batch, s_length_batch = tf.train.shuffle_batch([sentence, s_length],
                                                                batch_size=hparams.batch_size,
                                                                capacity=2*hparams.batch_size,
                                                                min_after_dequeue=hparams.batch_size,
                                                                allow_smaller_final_batch=False)

        return sentence_batch, s_length_batch, num_of_examples


def main(data_path, batch_size, num_epochs):
    vocab_en, vocab_de = load_vocabs(data_path)
    hparams1, hparams2 = get_hparams(vocab_en, vocab_de, batch_size, num_epochs + 1)

    sentence1, sentence_length1, num1 = get_data(data_path, name='test_en', hparams=hparams1)
    sentence2, sentence_length2, num2 = get_data(data_path, name='test_de', hparams=hparams2)

    sampling_probability = tf.placeholder(tf.float32, shape=[])
    global_step = tf.Variable(0, name='global_step', dtype=tf.int32, trainable=False)

    den1, den2, back1, back2 = get_models(hparams1, hparams2, sampling_probability,
                                          inputs=(sentence1,
                                                  sentence2,
                                                  sentence_length1,
                                                  sentence_length2),
                                          use_denoising=True,
                                          global_step=global_step)

    with tf.Session() as sess:
        if os.path.isfile(f'{data_path}/model/nmt_model_epoch_current.ckpt.meta'):
            saver = tf.train.Saver()
            saver.restore(sess, f'{data_path}/model/nmt_model_epoch_current.ckpt')
            print('Model restored.')
        else:
            saver = tf.train.Saver()
            tf.global_variables_initializer().run()
            print('Model created.')

        tf.local_variables_initializer().run()

        all_vars = tf.trainable_variables()
        [tf.summary.histogram(var.name.split(':')[0], var) for var in all_vars]

        tf.summary.scalar('loss_den1', den1.loss, family='losses')
        tf.summary.scalar('loss_den2', den2.loss, family='losses')
        tf.summary.scalar('loss_back1', back1.loss, family='losses')
        tf.summary.scalar('loss_back2', back2.loss, family='losses')
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        
        merged = tf.summary.merge_all()
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        log_dir = f'{data_path}/output'
        writer = tf.summary.FileWriter(log_dir, sess.graph)
        saver = tf.train.Saver()

        num_of_iterations = (num1 // (2 * hparams1.batch_size)) + 1

        alfa = 1 + 0.2
        beta = 0.0007

        for epoch in range(num_epochs):
            samp_prob = 1 - epoch / (alfa * num_epochs)
            inv_samp_prob = 1 - samp_prob

            for batch_num in range(num_of_iterations):
                print('Running Denoising. ', end='')
                if batch_num % 40 == 0:
                    ress = tuple(sess.run([den1.loss, den2.loss,
                                           den1.train_op, den2.train_op,
                                           global_step, merged], feed_dict={sampling_probability: inv_samp_prob}))

                    writer.add_summary(summary=ress[-1], global_step=ress[-2])
                else:
                    ress = tuple(sess.run([den1.loss, den2.loss,
                                           den1.train_op, den2.train_op,
                                           global_step], feed_dict={sampling_probability: inv_samp_prob}))

                print(f'ep{epoch}: {ress[0]:.4f}, {ress[1]:.4f}', end=' ')

                print('Running Backtranslation. ', end='')
                if batch_num % 40 == 0:
                    ress = tuple(sess.run([back1.loss, back2.loss,
                                           back1.train_op, back2.train_op,
                                           global_step, merged], feed_dict={sampling_probability: inv_samp_prob}))

                    writer.add_summary(summary=ress[-1], global_step=ress[-2])
                else:
                    ress = tuple(sess.run([back1.loss, back2.loss,
                                           back1.train_op, back2.train_op,
                                           global_step], feed_dict={sampling_probability: inv_samp_prob}))

                print(f'ep{epoch}: {ress[0]:.4f}, {ress[1]:.4f}')

                samp_prob = samp_prob / (beta * samp_prob + 1)
                inv_samp_prob = 1 - samp_prob

                if batch_num % 80 == 0 and batch_num != 0:
                    save_path = saver.save(sess, f'{data_path}/model/nmt_model_epoch_current.ckpt')
                    print('Model saved in path: %s' % save_path)

            save_path = saver.save(sess, f'{data_path}/model/nmt_model_epoch.ckpt')
            print('Model saved in path: %s' % save_path)

        coord.request_stop()
        coord.join(threads)

        writer.close()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        data_path = '/home/acivgin/PycharmProjects/M2-DS-Internship/Data'
    main(data_path, batch_size=10, num_epochs=2)
