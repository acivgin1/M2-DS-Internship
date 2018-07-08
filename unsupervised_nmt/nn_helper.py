import os

import numpy as np
import tensorflow as tf

# from fixed_decode import fixed_decode

### lan1 - lan1_embedder -- (embeddings)                                     decoder1  (embeddings) - lan1
###                                       encoder -- (meaning) -- attention
### lan2 - lan2_embedder -- (embeddings)                                     decoder2  (embeddings) - lan2


class LanguageEncoderDecoder():
    def __init__(self, hparams, lan_encoder_decoder=None):
        if isinstance(lan_encoder_decoder, LanguageEncoderDecoder):
            self.lan_encoder_decoder = lan_encoder_decoder
        else:
            self.lan_encoder_decoder = None
        self.hparams = hparams

        self.sts_token = (self.hparams.sts_token_id).astype(np.int32)
        self.end_token = (self.hparams.end_token_id).astype(np.int32)
        self.unk_token = (self.hparams.unk_token_id).astype(np.int32)
        self.pad_token = (self.hparams.end_token_id).astype(np.int32)

        self.embedding_encoder = None
        self.embedding_decoder = None
        self.batch_size = self.hparams.batch_size
        self.name = hparams.name


    def embedder(self, _input):
        # with tf.variable_scope(f'{self.name}_embedder', reuse=tf.AUTO_REUSE):
        self.embedding_encoder = tf.get_variable(self.hparams.name,
                                                 [self.hparams.vocab_size, self.hparams.embed_size])

        # we share the same encoder and decoder for sentences
        self.embedding_decoder = self.embedding_encoder

        encoder_emb_inp = tf.nn.embedding_lookup(self.embedding_encoder, _input)
        return encoder_emb_inp

    # def language_mapper(self, _input):
    #     output = tf.contrib.layers.fully_connected(_input, self.hparams.size, activation=tf.nn.relu)
    #     return output

    def encoder(self, _input, _input_sequence_length, reuse=False):
        if self.lan_encoder_decoder is None:
            with tf.variable_scope(f'{self.name}_encoder', reuse=tf.AUTO_REUSE):
                self.fw_cell = tf.nn.rnn_cell.GRUCell(num_units=self.hparams.num_units // 2, reuse=reuse, name=f'{self.name}_fw')
                self.bw_cell = tf.nn.rnn_cell.GRUCell(num_units=self.hparams.num_units // 2, reuse=reuse, name=f'{self.name}_bw')

                scope = f'{self.name}_bidir_rnn'
                outputs, output_states = tf.nn.bidirectional_dynamic_rnn(self.fw_cell,
                                                                         self.bw_cell,
                                                                         _input,
                                                                         sequence_length=_input_sequence_length,
                                                                         dtype=tf.float32,
                                                                         scope=scope)
                return tf.concat(outputs, axis=2), tf.concat(output_states, axis=1)
        else:
            outputs, output_states = self.lan_encoder_decoder.encoder(_input, _input_sequence_length, reuse=True)

        return outputs, output_states

    def decoder(self, _input_states, _input_sequence_length, mode, _encoder_inputs=None):
        ### internal functions start
        def _create_attention_mechanism(_encoder_output_states, _input_sequence_length, hparams):
            # attention_states = tf.transpose(_encoder_output_states, [1, 0, 2])

            attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=hparams.num_units,
                                                                    memory=_encoder_output_states,
                                                                    memory_sequence_length=_input_sequence_length,
                                                                    dtype=tf.float32,
                                                                    name=f'{self.name}_luong_att')
            return attention_mechanism

        def _wrap_decoder_cell(decoder_cell, attention_mechanism, hparams):
            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell,
                                                               attention_mechanism,
                                                               attention_layer_size=hparams.num_units,
                                                               name=f'{self.name}_att_wrapper')
            return decoder_cell
        ### internal functions end
        with tf.variable_scope(f'{self.name}_decoder', reuse=tf.AUTO_REUSE):
            with tf.variable_scope(f'{self.name}_decoder_cell', reuse=tf.AUTO_REUSE):
                decoder_cell = tf.nn.rnn_cell.GRUCell(num_units=self.hparams.num_units)
            with tf.variable_scope(f'{self.name}_attention_mech', reuse=tf.AUTO_REUSE):
                attention_mechanism = _create_attention_mechanism(_input_states, _input_sequence_length, self.hparams)

            decoder_cell = _wrap_decoder_cell(decoder_cell, attention_mechanism, self.hparams)
            decoder_initial_state = decoder_cell.zero_state(self.batch_size, dtype=tf.float32).clone(
                cell_state=_input_states[:, -1, :])

            projection_layer = None
            if mode == 'denoising' or mode == 'backtranslation_main':
                if _encoder_inputs is None:
                    raise ValueError(f"Only call 'decoder' in '{mode}' mode with provided _encoder_inputs.")
                # the _encoder_inputs are the ground truth for denoising and for the main language model
                with tf.variable_scope(f'{self.name}_training_helper', reuse=tf.AUTO_REUSE):
                    helper = tf.contrib.seq2seq.TrainingHelper(_encoder_inputs, _input_sequence_length)

                projection_layer = tf.layers.Dense(self.hparams.vocab_size, use_bias=False)

            elif mode == 'backtranslation_sec':
                # secondary decoder should behave freely so we treat it as if its in inference mode
                with tf.variable_scope(f'{self.name}_greedy_embedding_helper', reuse=tf.AUTO_REUSE):
                    start_tokens = tf.fill([self.batch_size], self.sts_token)
                    helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.embedding_decoder,
                                                                      start_tokens,
                                                                      self.end_token)
            else:
                raise ValueError("Only call 'decoder' with mode set to 'denoising', 'back_translation_main' or "
                                 "'back_translation_sec'.")

            decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell,
                                                      helper,
                                                      decoder_initial_state,
                                                      output_layer=projection_layer)
            outputs, _, output_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder=decoder,
                                                                                    output_time_major=False,
                                                                                    impute_finished=True,
                                                                                    maximum_iterations=self.hparams.max_out_length)
            return outputs, output_sequence_lengths

    def loss(self, _prevent_positional=None, labels=None, logits=None, lan1_meaning=None, lan2_meaning=None):
        if _prevent_positional is not None:
            raise  ValueError("Only call 'loss' with named arguments (labels=..., logits=...,)")

        if (lan1_meaning is not None) != (lan2_meaning is not None):
            raise ValueError("Only call 'loss' with both meanings or neither.")

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)

        if lan1_meaning is not None and lan2_meaning is not None:
            meaning_se = tf.losses.mean_squared_error(lan1_meaning, lan2_meaning)
        else:
            meaning_se = tf.zeros(1)

        train_loss = tf.reduce_sum(cross_entropy) + tf.reduce_mean(meaning_se)
        return train_loss / self.batch_size


    def shuffle_words(self, _input):
        # so the random_shuffle shuffles only the first dimension so we need to bring the words to the front
        # shuffle them and bring them back where they were.
        with tf.name_scope(f'{self.name}_shuffle_words'):
            shuffled = tf.random_shuffle(tf.transpose(_input, [1, 0, 2]))
            return tf.transpose(shuffled, [1, 0, 2])

    def denoising_model(self, _input, _input_sequence_length):
        '''
        Denoising_model is used for denoising training, when the decoder has to decode the input provided to the encoder
        :param _input: An input sequence to be encoded and then decoded, 2dim tensor
        :param _input_sequence_length: Input sequence length, tensor of dimensions [batch_size]
        :return:
        '''
        name = f'{self.name}_denoising'
        with tf.name_scope(f'{name}_model'):

            _embedded_input = self.embedder(_input=_input)

            # _noised_embedded_input = self.shuffle_words(_embedded_input)
            _noised_embedded_input = _embedded_input

            _output, _output_state = self.encoder(_input=_noised_embedded_input,
                                                  _input_sequence_length=_input_sequence_length)

            _output, _ = self.decoder(_input_states=_output,
                                      _input_sequence_length=_input_sequence_length,
                                      mode='denoising',
                                      _encoder_inputs=_embedded_input)

            loss = self.loss(labels=_input, logits=_output[0])

            return _output, loss

    def backtranslation_model(self, _input, _input_sequence_length, lan_enc_dec):
        name = f'{self.name}_backtrans'
        with tf.name_scope(f'{name}_model'):
            _embedded_input = self.embedder(_input=_input)

            _outputs, _output_state = self.encoder(_input=_embedded_input,
                                                   _input_sequence_length=_input_sequence_length)

            lan1_meaning = tf.concat((_outputs[:, -1, :], _output_state), axis=0)

            _outputs, _output_sequence_lengths = lan_enc_dec.decoder(_input_states=_outputs,
                                                                     _input_sequence_length=_input_sequence_length,
                                                                     mode='backtranslation_sec')

            _embedded_input = lan_enc_dec.embedder(_input=_outputs[1])

            _outputs, _output_state = lan_enc_dec.encoder(_input=_embedded_input,
                                                          _input_sequence_length=_output_sequence_lengths)
            lan2_meaning = tf.concat((_outputs[:, -1, :], _output_state), axis=0)


            _outputs, _ = self.decoder(_input_states=_outputs,
                                       _input_sequence_length=_output_sequence_lengths,
                                       mode='backtranslation_main',
                                       _encoder_inputs=_embedded_input)

            loss = self.loss(labels=_input, logits=_outputs[0], lan1_meaning=lan1_meaning, lan2_meaning=lan2_meaning)
            return _outputs, loss


if __name__ == '__main__':
    cur_path = os.path.dirname(__file__)
    data_path = os.path.relpath('../Data', cur_path)
    lan1_sents = np.load(f'{data_path}/test-full/en_sents.npy').astype(np.int32)
    lan2_sents = np.load(f'{data_path}/test-full/de_sents.npy').astype(np.int32)

    voc1_size = (lan1_sents.max() + 1).astype(np.int32)
    voc2_size = (lan2_sents.max() + 1).astype(np.int32)

    lan1_sents = np.split(lan1_sents, lan1_sents.shape[0])
    lan1_sents = [np.squeeze(x) for x in lan1_sents]

    lan2_sents = np.split(lan2_sents, lan2_sents.shape[0])
    lan2_sents = [np.squeeze(x) for x in lan2_sents]

    hparams1 = tf.contrib.training.HParams(
        embed_size=300,
        batch_size=32,
        vocab_size=voc1_size,
        num_units=100,
        max_out_length=50,
        end_token_id=voc1_size - 1,
        sts_token_id=voc1_size - 2,
        unk_token_id=voc1_size - 3,
        name='lan1')

    hparams2 = tf.contrib.training.HParams(
        embed_size=300,
        batch_size=32,
        vocab_size=voc2_size,
        num_units=100,
        max_out_length=50,
        end_token_id=voc2_size - 1,
        sts_token_id=voc2_size - 2,
        unk_token_id=voc2_size - 3,
        name='lan2')

    lan1_enc_dec = LanguageEncoderDecoder(hparams1)
    lan2_enc_dec = LanguageEncoderDecoder(hparams2, lan1_enc_dec)

    _input_lan1 = tf.placeholder(tf.int32, shape=(hparams1.batch_size, hparams1.max_out_length))
    _input_lan2 = tf.placeholder(tf.int32, shape=(hparams2.batch_size, hparams2.max_out_length))

    _input_sequence_length = tf.constant(50, dtype=tf.int32, shape=[hparams1.batch_size])

    output_den1, loss_den1 = lan1_enc_dec.denoising_model(_input_lan1, _input_sequence_length)
    with tf.variable_scope(tf.get_variable_scope(), reuse=False) as scope:
        optimizer_den1 = tf.train.AdamOptimizer().minimize(loss_den1)

    output_den2, loss_den2 = lan2_enc_dec.denoising_model(_input_lan2, _input_sequence_length)
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE) as scope:
        optimizer_den2 = tf.train.AdamOptimizer().minimize(loss_den2)
    #
    output_back1, loss_back1 = lan1_enc_dec.backtranslation_model(_input_lan1, _input_sequence_length, lan2_enc_dec)
    with tf.variable_scope(tf.get_variable_scope(), reuse=True) as scope:
        optimizer_back1 = tf.train.AdamOptimizer().minimize(loss_back1)
    #
    output_back2, loss_back2 = lan2_enc_dec.backtranslation_model(_input_lan2, _input_sequence_length, lan1_enc_dec)
    with tf.variable_scope(tf.get_variable_scope(), reuse=True) as scope:
        optimizer_back2 = tf.train.AdamOptimizer().minimize(loss_back2)


    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        writer = tf.summary.FileWriter('output', sess.graph)
        loss1, loss2 = sess.run([loss_den1, loss_den2, loss_back1, loss_back2],
                                      feed_dict={_input_lan1: lan1_sents[:hparams1.batch_size],
                                                 _input_lan2: lan2_sents[:hparams2.batch_size]})

        print(loss1)
        # print(loss2)
        writer.close()