import os
import sys
import time
import json

import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

from main_helper import *
from unmt_model import LanguageEncoderDecoder


G_HPARAMS = tf.contrib.training.HParams(
    # ADAM params
    learning_rate=1e-3*0.45,
    beta1=0.9,
    beta2=0.9999,
    epsilon=1e-4,
    gradient_clipping=False
)

def get_hparams(vocab1, vocab2, batch_size, num_epochs, num_of_examples):
    voc1_size = len(vocab1) + 3
    voc2_size = len(vocab2) + 3

    hparams1 = tf.contrib.training.HParams(
        embed_size=300,
        batch_size=batch_size,
        vocab_size=voc1_size,
        num_units=300,
        max_out_length=30,
        dtype=tf.float32,
        pad_token_id=np.array(voc1_size - 1, dtype=np.int32),
        end_token_id=np.array(voc1_size - 1, dtype=np.int32),
        sts_token_id=np.array(voc1_size - 2, dtype=np.int32),
        unk_token_id=np.array(voc1_size - 3, dtype=np.int32),
        name='en',
        num_epochs=num_epochs,
        num_of_examples=num_of_examples)

    hparams2 = tf.contrib.training.HParams(
        embed_size=hparams1.embed_size,
        batch_size=hparams1.batch_size,
        vocab_size=voc2_size,
        num_units=hparams1.num_units,
        max_out_length=hparams1.max_out_length,
        dtype=hparams1.dtype,
        pad_token_id=np.array(voc2_size - 1, dtype=np.int32),
        end_token_id=np.array(voc2_size - 1, dtype=np.int32),
        sts_token_id=np.array(voc2_size - 2, dtype=np.int32),
        unk_token_id=np.array(voc2_size - 3, dtype=np.int32),
        name='de',
        num_epochs=num_epochs,
        num_of_examples=num_of_examples)
    
    return hparams1, hparams2


def main(data_path, logs_path, batch_size, num_epochs, num_of_examples):
    vocab_en, vocab_de = load_vocabs(data_path)
    hparams1, hparams2 = get_hparams(vocab_en, vocab_de, batch_size, num_epochs * 2, num_of_examples)

    sentence1, sentence_length1, num1 = get_data(data_path, name='dummy_en', hparams=hparams1)
    sentence2, sentence_length2, num2 = get_data(data_path, name='dummy_de', hparams=hparams2)

    sampling_probability = tf.placeholder(tf.float16, shape=[])
    global_step = tf.Variable(0, name='global_step', dtype=tf.int32, trainable=False)
    write_step = tf.Variable(0, name='write_step', dtype=tf.int32, trainable=False)
    
    den1, den2, back1, back2 = get_models(hparams1, hparams2, G_HPARAMS, sampling_probability,
                                          inputs=(sentence1,
                                                  sentence2,
                                                  sentence_length1,
                                                  sentence_length2),
                                          use_denoising=True,
                                          global_step=global_step)
    
    en2de_output, en2de_length = den1.model.translation(sentence1, sentence_length1, den2.model)
    de2en_output, de2en_length = den2.model.translation(sentence2, sentence_length2, den1.model)

    with tf.Session() as sess:
        if os.path.isfile(f'{logs_path}/nmt_model_epoch_current.ckpt.meta'):
            saver = tf.train.Saver()
            saver.restore(sess, f'{logs_path}/nmt_model_epoch_current.ckpt')
            print('Latest model restored.')
        elif os.path.isfile(f'{logs_path}/nmt_model_epoch.ckpt.meta'):
            saver = tf.train.Saver()
            saver.restore(sess, f'{logs_path}/nmt_model_epoch.ckpt')
            print('Stable model restored.')     
        else:
            saver = tf.train.Saver()
            tf.global_variables_initializer().run()
            print('Model created.')
                  
        tf.local_variables_initializer().run()

        all_vars = tf.trainable_variables()
        [tf.summary.histogram(f'{var.name.split(":")[0]}_hist', var) for var in all_vars]

        tf.summary.scalar('loss_den1', den1.loss, family='1losses')
        tf.summary.scalar('loss_den2', den2.loss, family='1losses')
        tf.summary.scalar('loss_back1', back1.loss, family='1losses')
        tf.summary.scalar('loss_back2', back2.loss, family='1losses')
        
        tf.summary.scalar('samp_prob', sampling_probability)
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(logs_path, sess.graph)
        
        config = projector.ProjectorConfig()
        embedding1 = config.embeddings.add()
        embedding1.tensor_name = den1.model.embedding_encoder.name
        embedding1.metadata_path = f'{logs_path}/en.tsv'
        
        embedding2 = config.embeddings.add()
        embedding2.tensor_name = den2.model.embedding_encoder.name
        embedding2.metadata_path = f'{logs_path}/de.tsv'
        
        projector.visualize_embeddings(writer, config)
        
        
        num_of_iterations = (num1 // (hparams1.batch_size * 2))
        alfa = 0.8
        
        for epoch in range(num_epochs):
            samp_prob = 1 - np.square((alfa * epoch) / (num_epochs - 1))
            inv_samp_prob = 1 - samp_prob
            
            for batch_num in range(num_of_iterations):
                start_time = time.time()
                print(f'Running ep{epoch}/{num_epochs}-', end='')
                print(f'b{batch_num}/{num_of_iterations}-L1: ', end='')
                
                if False and batch_num % 2 == 0:
                    ress = tuple(sess.run([den1.loss, den2.loss,
                                           den1.train_op, den2.train_op,
                                           write_step, merged], feed_dict={sampling_probability: inv_samp_prob}))

                    writer.add_summary(summary=ress[-1], global_step=ress[-2])
                    write_step = write_step + 1
                else:
                      ress = tuple(sess.run([back1.loss, back1.train_op,
                                             global_step],
                                            feed_dict={sampling_probability: inv_samp_prob}))
                
                print(f'{time.time()-start_time:.2f}s: {ress[0]:.2f}', end=' ') # , {ress[1]:.4f}', end=' ')
                start_time = time.time()
                
                print('L2: ', end='')
                if False and batch_num % 2 == 0 and batch_num != 0:
                    ress = tuple(sess.run([back1.loss, back2.loss,
                                           back1.train_op, back2.train_op,
                                           write_step, merged], feed_dict={sampling_probability: inv_samp_prob}))
                    
                    writer.add_summary(summary=ress[-1], global_step=ress[-2])
                    write_step = write_step + 1
                else:
                    ress = tuple(sess.run([back2.loss, back2.train_op,
                                           global_step],
                                          feed_dict={sampling_probability: inv_samp_prob}))
                
                print(f'{time.time()-start_time:.2f}s: {ress[0]:.2f}') #  {ress[1]:.4f}')
                
                if batch_num % 100 == 0 and batch_num != 0:
                    # because epoch lasts for so long time, we need to save more often.
                    save_path = saver.save(sess, f'{logs_path}/nmt_model_epoch_current.ckpt')
                    print('Model saved in path: %s' % save_path)
                if batch_num % 25 == 0:
                    translated1, translated2, len1, len2, orig1, orig2 = sess.run([en2de_output, de2en_output,
                                                                                   en2de_length, de2en_length,
                                                                                   sentence1, sentence2],
                                                                                  feed_dict={sampling_probability: 0.0})
                    for i in range(1):
                        print(ids2wordlist(orig1[i, ...], vocab_en))
                        print(ids2wordlist(translated1[1][i, ...], vocab_de, len1[i]))

                        print(ids2wordlist(orig2[i, ...], vocab_de))
                        print(ids2wordlist(translated2[1][i, ...], vocab_en, len2[i]))   
                    
            # after one epoch we save the epoch model.
            save_path = saver.save(sess, f'{logs_path}/nmt_model_epoch.ckpt')
            print('Model saved in path: %s' % save_path)
        
        coord.request_stop()
        coord.join(threads)

        writer.close()

if __name__ == '__main__':
    data_path='/home/acivgin/PycharmProjects/M2-DS-Internship/Data'
    logs_path='/home/acivgin/PycharmProjects/M2-DS-Internship/logs'
    main(data_path, logs_path, batch_size=1, num_epochs=1001, num_of_examples=3800002)
