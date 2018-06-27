import tensorflow as tf

from .fixed_decode import fixed_decode
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

    def embedder(self, _input):
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
        if self.lan_encoder_decoder is not None:
            fw_cell = tf.nn.rnn_cell.GRUCell(num_units=self.hparams.num_units)
            bw_cell = tf.nn.rnn_cell.GRUCell(num_units=self.hparams.num_units)

            outputs, output_states = tf.contrib.rnn.bidirectional_dynamic_rnn(fw_cell,
                                                                              bw_cell,
                                                                              _input,
                                                                              sequence_length=_input_sequence_length)
        else:
            outputs, output_states = self.lan_encoder_decoder.encoder(_input, _input_sequence_length)

        return outputs[1], output_states[1]

    def decoder(self, _input_states, _input_sequence_length, mode, _encoder_inputs=None):
        ### internal functions start
        def _create_attention_mechanism(_encoder_output_states, _input_sequence_length, hparams):
            # attention_states = tf.transpose(_encoder_output_states, [1, 0, 2])

            attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=hparams.num_units,
                                                                    memory=_encoder_output_states,
                                                                    memory_sequence_length=_input_sequence_length)
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

        if mode == 'denoising' or mode == 'back_translation_main':
            if _encoder_inputs is None:
                raise ValueError("Only call 'decoder' in '{}' mode with provided _encoder_inputs.".format(mode))
            # the _encoder_inputs are the ground truth for denoising and for the main language model
            helper = tf.contrib.seq2seq.TrainingHelper(_encoder_inputs, _input_sequence_length)

        elif mode == 'back_translation_sec':
            # secondary decoder should behave freely so we treat it as if its in inference mode
            start_tokens = tf.fill([self.batch_size], [self.start_token])
            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.embedding_decoder,
                                                              start_tokens,
                                                              self.end_token)
        else:
            raise ValueError("Only call 'decoder' with mode set to 'denoising', 'back_translation_main' or "
                             "'back_translation_sec'.")

        decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell,
                                                  helper,
                                                  _input_states)
        outputs, _ = tf.contrib.seq2seq.dynamic_decode(decoder=decoder,
                                                       output_time_major=False,
                                                       impute_finished=True,
                                                       maximum_iterations=self.hparams.max_out_length)
        return outputs

    def loss(self, _prevent_positional=None, labels=None, logits=None, lan1_meaning=None, lan2_meaning=None):
        if _prevent_positional is not None:
            raise  ValueError("Only call 'loss' with named arguments (labels=..., logits=...,)")

        if lan1_meaning is not None != lan2_meaning is not None:
            raise ValueError("Only call 'loss' with both meanings or neither.")

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)

        if lan1_meaning is not None and lan2_meaning is not None:
            meaning_se = tf.losses.mean_squared_error(lan1_meaning, lan2_meaning)
        else:
            meaning_se = tf.zeros(1)

        train_loss = tf.reduce_mean(cross_entropy) + tf.reduce_mean(meaning_se)
        return train_loss / self.batch_size

    def language_encoder(self, _input, _input_sequence_length):
        _output = self.embedder(_input=_input)
        # _output = self.language_mapper(_input=_output)
        _output, _output_states = self.encoder(_input=_output, _input_sequence_length=_input_sequence_length)
        return _output, _output_states

    def denoising_model(self, _input, _input_sequence_length):
        '''
        Denoising_model is used for denoising training, when the decoder has to decode the input provided to the encoder
        :param _input: An input sequence to be encoded and then decoded, 2dim tensor
        :param _input_sequence_length: Input sequence length, tensor of dimensions [batch_size]
        :return:
        '''
        # TODO (acivgin): can we get some noise in this method
        _noised_input = _input
        _output, _output_states = self.language_encoder(_noised_input, _input_sequence_length)
        _output = self.decoder(_input_states=_output_states,
                               _input_sequence_length=_input_sequence_length,
                               mode='denoising',
                               _encoder_inputs=_input)
        return _output

    # def language_decoder(self, _input_states, _input_sequence_length, mode=None, _encoder_inputs=None):
    #     _output = self.decoder(_input_states=_input_states,
    #                            _input_sequence_length=_input_sequence_length,
    #                            mode=mode,
    #                            _encoder_inputs=_encoder_inputs)
    #     return _output

# def build_denoising_model