import os

import numpy as np
import tensorflow as tf
from tqdm import tqdm


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def write_all_data(data_path):
    def get_data(npz_filename):
        sents = np.load(npz_filename)

        lan_sents = sents['sentences'].astype(np.int32)
        lan_sent_lenghts = sents['lengths'].astype(np.int32)

        print(lan_sents.shape)
        print(lan_sent_lenghts.shape)

        return lan_sents[:1500000, :], lan_sent_lenghts[:1500000]

    def create_tfrecord(tfrecords_filename, lan_sents, lan_sent_lengths):
        writer = tf.python_io.TFRecordWriter(tfrecords_filename)

        for idx in tqdm(range(lan_sent_lenghts.size)):
            feature = {f'sentence': _bytes_feature(tf.compat.as_bytes(lan_sents[idx, :].tostring())),
                       f'length': _int64_feature(lan_sent_lengths[idx])}
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

        writer.close()

    names = [('test', 'en'), ('test', 'de'), ('train', 'en'), ('train', 'de')]
    for name in names[2:]:
        npz_filename = f'{data_path}/dataset_utils/{name[0]}_sents_{name[1]}.npz'
        tfrecords_filename = f'{data_path}/{name[0]}_{name[1]}1M5.tfrecords'

        if os.path.isfile(tfrecords_filename):
            print(f'{tfrecords_filename} already exists.')
            continue

        lan_sents, lan_sent_lenghts = get_data(npz_filename)
        create_tfrecord(tfrecords_filename, lan_sents, lan_sent_lenghts)


def read_tfrecord(data_path):
    tfrecords_filename = f'{data_path}/train_en.tfrecords'

    with tf.Session() as sess:
        feature = {'sentence': tf.FixedLenFeature([], tf.string),
                   'length': tf.FixedLenFeature([], tf.int64)}

        filename_queue = tf.train.string_input_producer([tfrecords_filename], num_epochs=1)

        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)

        features = tf.parse_single_example(serialized_example, features=feature)

        sentence = tf.decode_raw(features['sentence'], tf.int32)
        sentence = tf.reshape(sentence, [40])

        s_length = tf.cast(features['length'], tf.int32)

        sentence, s_length = tf.train.shuffle_batch([sentence, s_length],
                                                    batch_size=10,
                                                    capacity=30,
                                                    num_threads=1,
                                                    min_after_dequeue=20)

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        print(sentence.shape)
        print(s_length.shape)
        sentence, s_length = sess.run([sentence, s_length])
        print(f'{s_length}:\n{sentence}')

        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    cur_path = os.path.dirname(__file__)
    data_path = os.path.realpath(os.path.join(cur_path, '../../Data'))
    write_all_data(data_path)
    # read_tfrecord(data_path)
