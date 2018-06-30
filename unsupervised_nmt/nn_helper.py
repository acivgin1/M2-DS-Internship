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

        self.start_token = self.hparams.embed_size - 3
        self.end_token = self.hparams.embed_size - 2
        self.unk_token = self.hparams.embed_size - 1

        self.embedding_encoder = None
        self.embedding_decoder = None
        self.batch_size = self.hparams.batch_size

        self.fw_cell = tf.nn.rnn_cell.GRUCell(num_units=self.hparams.num_units // 2)
        self.bw_cell = tf.nn.rnn_cell.GRUCell(num_units=self.hparams.num_units // 2)

    def embedder(self, _input):
        with tf.variable_scope('embedder', reuse=tf.AUTO_REUSE):
            self.embedding_encoder = tf.get_variable(self.hparams.embed_name,
                                                     [self.hparams.vocab_size, self.hparams.embed_size])

        # we share the same encoder and decoder for sentences
        self.embedding_decoder = self.embedding_encoder

        encoder_emb_inp = tf.nn.embedding_lookup(self.embedding_encoder, _input)
        return encoder_emb_inp

    # def language_mapper(self, _input):
    #     output = tf.contrib.layers.fully_connected(_input, self.hparams.size, activation=tf.nn.relu)
    #     return output

    def encoder(self, _input, _input_sequence_length):
        if self.lan_encoder_decoder is None:
            with tf.variable_scope('encoder_cells', reuse=tf.AUTO_REUSE):
                outputs, output_states = tf.nn.bidirectional_dynamic_rnn(self.fw_cell,
                                                                         self.bw_cell,
                                                                         _input,
                                                                         sequence_length=_input_sequence_length,
                                                                         dtype=tf.float32)
        else:
            outputs, output_states = self.lan_encoder_decoder.encoder(_input, _input_sequence_length)

        return tf.concat(outputs, axis=2), tf.concat(output_states, axis=1)

    def decoder(self, _input_states, _input_sequence_length, mode, _encoder_inputs=None):
        ### internal functions start
        def _create_attention_mechanism(_encoder_output_states, _input_sequence_length, hparams):
            # attention_states = tf.transpose(_encoder_output_states, [1, 0, 2])

            attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=hparams.num_units,
                                                                    memory=_encoder_output_states,
                                                                    memory_sequence_length=_input_sequence_length,
                                                                    dtype=tf.float32)
            return attention_mechanism

        def _wrap_decoder_cell(decoder_cell, attention_mechanism, hparams):
            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell,
                                                               attention_mechanism,
                                                               attention_layer_size=hparams.num_units)
            return decoder_cell
        ### internal functions end

        decoder_cell = tf.nn.rnn_cell.GRUCell(num_units=self.hparams.num_units)
        attention_mechanism = _create_attention_mechanism(_input_states, _input_sequence_length, self.hparams)

        decoder_cell = _wrap_decoder_cell(decoder_cell, attention_mechanism, self.hparams)
        decoder_initial_state = decoder_cell.zero_state(self.batch_size, dtype=tf.float32).clone(
            cell_state=_input_states[:,-1,:])

        if mode == 'denoising' or mode == 'backtranslation_main':
            if _encoder_inputs is None:
                raise ValueError("Only call 'decoder' in '{}' mode with provided _encoder_inputs.".format(mode))
            # the _encoder_inputs are the ground truth for denoising and for the main language model
            helper = tf.contrib.seq2seq.TrainingHelper(_encoder_inputs, _input_sequence_length)

        elif mode == 'backtranslation_sec':
            # secondary decoder should behave freely so we treat it as if its in inference mode
            start_tokens = tf.fill([self.batch_size], self.start_token)
            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.embedding_decoder,
                                                              start_tokens,
                                                              self.end_token)
        else:
            raise ValueError("Only call 'decoder' with mode set to 'denoising', 'back_translation_main' or "
                             "'back_translation_sec'.")

        decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell,
                                                  helper,
                                                  decoder_initial_state)
        outputs, _, output_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder=decoder,
                                                                                output_time_major=False,
                                                                                impute_finished=True,
                                                                                maximum_iterations=self.hparams.max_out_length)
        return outputs, output_sequence_lengths

    def loss(self, _prevent_positional=None, labels=None, logits=None, lan1_meaning=None, lan2_meaning=None):
        if _prevent_positional is not None:
            raise  ValueError("Only call 'loss' with named arguments (labels=..., logits=...,)")

        if lan1_meaning is not None != lan2_meaning is not None:
            raise ValueError("Only call 'loss' with both meanings or neither.")
        # TODO: these will be embedded so sparse softmax wont do it
        # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        cross_entropy = tf.losses.mean_squared_error(labels=labels, logits=logits)

        if lan1_meaning is not None and lan2_meaning is not None:
            meaning_se = tf.losses.mean_squared_error(lan1_meaning, lan2_meaning)
        else:
            meaning_se = tf.zeros(1)

        train_loss = tf.reduce_mean(cross_entropy) + tf.reduce_mean(meaning_se)
        return train_loss / self.batch_size


    def denoising_model(self, _input, _input_sequence_length):
        '''
        Denoising_model is used for denoising training, when the decoder has to decode the input provided to the encoder
        :param _input: An input sequence to be encoded and then decoded, 2dim tensor
        :param _input_sequence_length: Input sequence length, tensor of dimensions [batch_size]
        :return:
        '''
        # TODO (acivgin): can we get some noise in this method
        _embedded_input = self.embedder(_input=_input)
        _noised_embedded_input = _embedded_input
        _output, _output_state = self.encoder(_input=_noised_embedded_input, _input_sequence_length=_input_sequence_length)

        _output, _ = self.decoder(_input_states=_output,
                                  _input_sequence_length=_input_sequence_length,
                                  mode='denoising',
                                  _encoder_inputs=_embedded_input)
        return _output

    def backtranslation_model(self, _input, _input_sequence_length, lan_enc_dec):
        _embedded_input = self.embedder(_input=_input)
        # lan_enc_dec =
        _output, _output_state = self.encoder(_input=_embedded_input, _input_sequence_length=_input_sequence_length)

        _output, _output_sequence_lengths = lan_enc_dec.decoder(_input_states=_output,
                                                                _input_sequence_length=_input_sequence_length,
                                                                mode='backtranslation_sec')

        _embedded_input = self.embedder(_input=_output[1])

        _output, _output_state = lan_enc_dec.encoder(_input=_embedded_input, _input_sequence_length=_output_sequence_lengths)

        _output, _ = self.decoder(_input_states=_output,
                                  _input_sequence_length=_output_sequence_lengths,
                                  mode='backtranslation_main',
                                  _encoder_inputs=_embedded_input)
        return _output


def create_train_model(hparams1, hparams2):
    lan1_enc_dec = LanguageEncoderDecoder(hparams1)
    lan2_enc_dec = LanguageEncoderDecoder(hparams2, lan1_enc_dec)


if __name__ == '__main__':
    hparams1 = tf.contrib.training.HParams(
        embed_size=300,
        batch_size=3,
        embed_name='embedder1',
        vocab_size=2000,
        num_units=200,
        max_out_length=10)

    hparams2 = tf.contrib.training.HParams(
        embed_size=300,
        batch_size=3,
        embed_name='embedder2',
        vocab_size=2000,
        num_units=200,
        max_out_length=10)

    lan1_enc_dec = LanguageEncoderDecoder(hparams1)

    lan2_enc_dec = LanguageEncoderDecoder(hparams2, lan1_enc_dec)
    _input = tf.placeholder(tf.int32, shape=(hparams1.batch_size, 10))
    _input_sequence_length = tf.constant([10, 8, 7], dtype=tf.int32, shape=[3])

    output_1 = lan1_enc_dec.denoising_model(_input=_input, _input_sequence_length=_input_sequence_length)
    output_2 = lan2_enc_dec.denoising_model(_input=_input, _input_sequence_length=_input_sequence_length)

    output = lan1_enc_dec.backtranslation_model(_input, _input_sequence_length, lan2_enc_dec)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        writer = tf.summary.FileWriter("output", sess.graph)
        print(sess.run([output_1, output_2, output], feed_dict={_input: [[1, 2, 3, 4, 5, 6, 7, 8, 9, 1],
                                                                         [3, 4, 7, 5, 6, 1, 2, 5, 8, 10],
                                                                         [3, 4, 7, 5, 6, 1, 2, 5, 8, 10]]}))
        writer.close()