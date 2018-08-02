import tensorflow as tf


class LanguageEncoderDecoder():
    def __init__(self, hparams, lan_encoder_decoder=None):
        if isinstance(lan_encoder_decoder, LanguageEncoderDecoder):
            self.lan_encoder_decoder = lan_encoder_decoder
        else:
            self.lan_encoder_decoder = None
        self.hparams = hparams

        self.sts_token = self.hparams.sts_token_id
        self.end_token = self.hparams.end_token_id
        self.unk_token = self.hparams.unk_token_id
        self.pad_token = self.hparams.pad_token_id

        self.embedding_encoder = None
        self.embedding_decoder = None
        self.embedding_name = None
        self.projection_layer = None

        self.fw_cell = None
        self.bw_cell = None
        self.decoder_cell = None

        self.batch_size = self.hparams.batch_size
        self.name = hparams.name

    def embedder(self, _input):
        if self.embedding_encoder is None:
            with tf.variable_scope(f'{self.name}_embedder/', reuse=tf.AUTO_REUSE):
                self.embedding_name = f'{self.name}_acc_embedder/'
                self.embedding_encoder = tf.get_variable(self.embedding_name,
                                                         [self.hparams.vocab_size, self.hparams.embed_size])
            self.embedding_decoder = self.embedding_encoder

        encoder_emb_inp = tf.nn.embedding_lookup(self.embedding_encoder, _input)
        return encoder_emb_inp

    def encoder(self, _input, _input_sequence_length):
        if self.lan_encoder_decoder is None:
            paddings = tf.constant([[0, 0], [0, 1], [0, 0]])
            _input = tf.pad(_input[:, 1:, :], paddings, 'CONSTANT', constant_values=self.pad_token)

            with tf.variable_scope(f'{self.name}_encoder/', reuse=tf.AUTO_REUSE):
                if self.fw_cell is None:
                    self.fw_cell = tf.nn.rnn_cell.GRUCell(num_units=self.hparams.num_units // 2, reuse=tf.AUTO_REUSE,
                                                          name=f'{self.name}_fw/')
                    self.bw_cell = tf.nn.rnn_cell.GRUCell(num_units=self.hparams.num_units // 2, reuse=tf.AUTO_REUSE,
                                                          name=f'{self.name}_bw/')

                scope = f'{self.name}_bidir_rnn/'
                outputs, output_states = tf.nn.bidirectional_dynamic_rnn(self.fw_cell,
                                                                         self.bw_cell,
                                                                         _input,
                                                                         sequence_length=_input_sequence_length,
                                                                         dtype=tf.float32,
                                                                         scope=scope)
                return tf.concat(outputs, axis=2), tf.concat(output_states, axis=1)
        else:
            outputs, output_states = self.lan_encoder_decoder.encoder(_input, _input_sequence_length)

        return outputs, output_states

    def decoder(self, _input_states, _last_state, _input_sequence_length, mode, _encoder_inputs=None,
                sampling_probability=None):
        ### internal functions start
        def _create_attention_mechanism(_encoder_output_states, _input_sequence_length, hparams):
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=hparams.num_units,
                                                                    memory=_encoder_output_states,
                                                                    memory_sequence_length=_input_sequence_length,
                                                                    dtype=tf.float32,
                                                                    name=f'{self.name}_luong_att/')
            return attention_mechanism

        def _wrap_decoder_cell(decoder_cell, attention_mechanism, hparams):
            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell,
                                                               attention_mechanism,
                                                               attention_layer_size=hparams.num_units,
                                                               name=f'{self.name}_att_wrapper/')
            return decoder_cell

        ### internal functions end
        if self.decoder_cell is None:
            with tf.variable_scope(f'{self.name}_decoder/', reuse=tf.AUTO_REUSE):
                with tf.variable_scope(f'{self.name}_decoder_cell/', reuse=tf.AUTO_REUSE):
                    self.decoder_cell = tf.nn.rnn_cell.GRUCell(num_units=self.hparams.num_units)
                with tf.variable_scope(f'{self.name}_attention_mech/', reuse=tf.AUTO_REUSE):
                    attention_mechanism = _create_attention_mechanism(_input_states, _input_sequence_length,
                                                                      self.hparams)

                self.decoder_cell = _wrap_decoder_cell(self.decoder_cell, attention_mechanism, self.hparams)

        with tf.variable_scope(f'{self.name}_basic_decoder/', reuse=tf.AUTO_REUSE):
            decoder_initial_state = self.decoder_cell.zero_state(self.batch_size, dtype=tf.float32).clone(
                cell_state=_last_state)

            projection_layer = None
            if mode == 'denoising' or mode == 'backtranslation_main':
                if _encoder_inputs is None:
                    raise ValueError(f"Only call 'decoder' in '{mode}' mode with provided _encoder_inputs.")
                # the _encoder_inputs are the ground truth for denoising and for the main language model
                with tf.variable_scope(f'{self.name}_training_helper/', reuse=tf.AUTO_REUSE):
                    helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(_encoder_inputs,
                                                                                 _input_sequence_length,
                                                                                 sampling_probability=sampling_probability,
                                                                                 embedding=self.embedding_decoder)

                projection_layer = tf.layers.Dense(self.hparams.vocab_size, use_bias=False)
                self.projection_layer = projection_layer

            elif mode == 'backtranslation_sec' or mode == 'inference':
                # secondary decoder should behave freely so we treat it as if its in inference mode
                with tf.variable_scope(f'{self.name}_greedy_embedding_helper/', reuse=tf.AUTO_REUSE):
                    start_tokens = tf.fill([self.batch_size], self.sts_token)
                    helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.embedding_decoder,
                                                                      start_tokens,
                                                                      self.end_token)
            else:
                raise ValueError("Only call 'decoder' with mode set to 'denoising', 'back_translation_main' or "
                                 "'back_translation_sec'.")

            if mode == 'inference':
                projection_layer = self.projection_layer

            decoder = tf.contrib.seq2seq.BasicDecoder(self.decoder_cell,
                                                      helper,
                                                      decoder_initial_state,
                                                      output_layer=projection_layer)
            outputs, _, output_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder=decoder,
                                                                                    output_time_major=False,
                                                                                    impute_finished=True,
                                                                                    maximum_iterations=self.hparams.max_out_length)
            return outputs, output_sequence_lengths

    def loss(self, _prevent_positional=None,
             labels=None, logits=None,
             lan1_meaning=None, lan2_meaning=None,
             target_sequence_length=None):

        if _prevent_positional is not None:
            raise ValueError("Only call 'loss' with named arguments (labels=..., logits=...,)")

        if (lan1_meaning is not None) != (lan2_meaning is not None):
            raise ValueError("Only call 'loss' with both meanings or neither.")

        paddings = tf.constant([[0, 0], [0, 1]])
        labels = tf.pad(labels[:, 1:], paddings, 'CONSTANT', constant_values=self.pad_token)

        logits_len = tf.shape(logits)[1]
        full_len = tf.shape(labels)[1]

        paddings = tf.Variable(tf.zeros(shape=(3, 2), dtype=tf.int32))
        paddings = paddings[1, 1].assign(full_len - logits_len)

        logits = tf.pad(logits, paddings, 'CONSTANT', constant_values=self.pad_token)

        target_weights = tf.sequence_mask(target_sequence_length - 1, self.hparams.max_out_length, dtype=logits.dtype)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)

        if lan1_meaning is not None and lan2_meaning is not None:
            meaning_se = tf.losses.mean_squared_error(lan1_meaning, lan2_meaning)
        else:
            meaning_se = tf.zeros(1)

        train_loss = tf.reduce_sum(cross_entropy * target_weights) + tf.reduce_sum(meaning_se)
        return train_loss / tf.to_float(tf.shape(labels)[0])

    def denoising_model(self, _input, _input_sequence_length, sampling_probability):
        name = f'{self.name}_denoising'
        with tf.name_scope(f'{name}_model/'):
            _embedded_input = self.embedder(_input=_input)
            _noised_embedded_input = _embedded_input

            _output, _output_state = self.encoder(_input=_noised_embedded_input,
                                                  _input_sequence_length=_input_sequence_length)

            #             with tf.name_scope(f'{name}_last_state'):
            #                 mask = tf.one_hot(_input_sequence_length - 1, depth=tf.shape(_outputs)[1], dtype=tf.bool, on_value=True, off_value=False)
            #                 last_outputs = tf.boolean_mask(_outputs, mask)
            #                 lan1_meaning = tf.concat((last_outputs, _output_state), axis=0)

            _output, _ = self.decoder(_input_states=_output,
                                      _input_sequence_length=_input_sequence_length,
                                      _last_state=_output_state,
                                      mode='denoising',
                                      _encoder_inputs=_embedded_input,
                                      sampling_probability=sampling_probability)

            loss = self.loss(labels=_input, logits=_output[0],
                             target_sequence_length=_input_sequence_length)

            return _output, loss

    def backtranslation_model(self, _input, _input_sequence_length, sampling_probability, lan_enc_dec,
                              translation=False):
        name = f'{self.name}_backtrans'
        with tf.name_scope(f'{name}_model/'):
            _embedded_input = self.embedder(_input=_input)

            _outputs, _output_state = self.encoder(_input=_embedded_input,
                                                   _input_sequence_length=_input_sequence_length)

            with tf.name_scope(f'{name}_first_meaning/'):
                mask = tf.one_hot(_input_sequence_length - 1, depth=tf.shape(_outputs)[1], dtype=tf.bool, on_value=True,
                                  off_value=False)
                lan1_meaning = tf.boolean_mask(_outputs, mask)
                # lan1_meaning = tf.concat((last_outputs, _output_state), axis=0)

            _outputs, _output_sequence_length = lan_enc_dec.decoder(_input_states=_outputs,
                                                                    _input_sequence_length=_input_sequence_length,
                                                                    _last_state=_output_state,
                                                                    mode='backtranslation_sec')

            paddings = tf.constant([[0, 0], [1, 0]])
            _padded_output = tf.pad(_outputs[1][:, :-1], paddings, 'CONSTANT', constant_values=self.sts_token)

            _embedded_input = lan_enc_dec.embedder(_input=_padded_output)

            _outputs, _output_state = lan_enc_dec.encoder(_input=_embedded_input,
                                                          _input_sequence_length=_output_sequence_length)

            with tf.name_scope(f'{name}_second_meaning/'):
                mask = tf.one_hot(_output_sequence_length - 1, depth=tf.shape(_outputs)[1], dtype=tf.bool,
                                  on_value=True, off_value=False)
                lan2_meaning = tf.boolean_mask(_outputs, mask)
                # lan2_meaning = tf.concat((last_outputs, _output_state), axis=0)

            _outputs, _ = self.decoder(_input_states=_outputs,
                                       _input_sequence_length=_output_sequence_length,
                                       _last_state=_output_state,
                                       mode='backtranslation_main',
                                       _encoder_inputs=_embedded_input,
                                       sampling_probability=sampling_probability)

            loss = self.loss(labels=_input, logits=_outputs[0],
                             lan1_meaning=lan1_meaning, lan2_meaning=lan2_meaning,
                             target_sequence_length=_input_sequence_length)

            return _outputs, loss

    def translation(self, _input, _input_sequence_length, lan_enc_dec):
        name = f'{self.name}_backtrans'
        with tf.name_scope(f'{name}_model/'):
            _embedded_input = self.embedder(_input=_input)
            _outputs, _output_state = self.encoder(_input=_embedded_input,
                                                   _input_sequence_length=_input_sequence_length)

            _outputs, _output_sequence_lengths = lan_enc_dec.decoder(_input_states=_outputs,
                                                                     _input_sequence_length=_input_sequence_length,
                                                                     _last_state=_output_state,
                                                                     mode='inference')
            return _outputs, _output_sequence_lengths