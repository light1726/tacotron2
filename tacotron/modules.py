import tensorflow as tf
import collections


class TacotronEncoder(tf.keras.Model):
    def __init__(self, input_dim, emb_size, n_conv, conv_kernel, conv_filter,
                 lstm_hidden, is_training, name='text_encoder', **kwargs):
        super(TacotronEncoder, self).__init__(name=name, **kwargs)
        self.emb_layer = tf.keras.layers.Embedding(
            input_dim=input_dim, output_dim=emb_size, name='text_embedding')
        self.conv_stack = []
        self.batch_norm_stack = []
        for i in range(n_conv):
            conv = tf.keras.layers.Conv1D(filters=conv_filter,
                                          kernel_size=conv_kernel,
                                          padding='SAME',
                                          activtion=None,
                                          name='conv_{}'.format(i))
            self.conv_stack.append(conv)
            batch_norm = tf.keras.layers.BatchNormalization(
                name='batch_norm_{}'.format(i))
            self.batch_norm_stack.append(batch_norm)
        self.blstm_layer = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units=lstm_hidden, return_sequences=True),
            merge_mode='concat', name='blstm_layer')
        self.training = is_training

    def call(self, inputs):
        # 1. text embeddings
        embs = self.emb_layer(inputs)
        conv_out = embs
        for conv, bn in zip(self.conv_stack, self.batch_norm_stack):
            conv_out = conv(conv_out)
            conv_out = bn(conv_out, training=self.training)
            conv_out = tf.nn.relu(conv_out)
        blstm_out = self.blstm_layer(conv_out)
        return blstm_out


class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, attention_dim, attention_window_size,
                 attention_filters, attention_kernel, cumulate_weight,
                 constraint_type, name='attention', **kwargs):
        """
        :param attention_dim:
        :param attention_window_size:
        :param attention_filters:
        :param attention_kernel:
        :param cumulate_weight:
        :param constraint_type: 'window' or 'monotonic' or None
        :param name:
        :param kwargs:
        """
        super(AttentionLayer, self).__init__(name=name, **kwargs)
        self.attention_dim = attention_dim
        self.query_layer = tf.keras.layers.Dense(
            units=attention_dim, use_bias=False, name='query_layer')
        self.memory_layer = tf.keras.layers.Dense(
            units=attention_dim, use_bias=False, name='memory_layer')
        self.location_convolution = tf.keras.layers.Conv1D(
            filters=attention_filters, kernel_size=attention_kernel,
            padding='same', activation=None, name='location_conv')
        self.location_layer = tf.keras.layers.Dense(
            units=attention_dim, use_bias=False, name='location_feature_layer')
        self.score_v = tf.compat.v1.get_variable(
            'attention_variable_projection', shape=[attention_dim, ])
        self.score_b = tf.compact.v1.get_variable(
            'attention_variable_bias', shape=[attention_dim, ],
            initializer='zeros')
        self.constraint_type = constraint_type
        self.window_size = attention_window_size
        self._cumulate = cumulate_weight

    def _location_sensitive_score(self, w_query, w_fil, w_keys):
        """
        :param w_query: [batch, 1, attention_dim]
        :param w_fil: [batch, max_time, attention_dim],
                processed previous alignments into location features
        :param w_keys: [batch, max_time, attention_dim]
        :return: [batch, max_time], attention score (energy)
        """
        return tf.math.reduce_sum(
            self.score_v * tf.nn.tanh(w_keys + w_query + w_fil + self.score_b),
            axis=2)

    def get_initial_alignments(self, batch_size, max_time, dtype=None):
        dtype = dtype if dtype is not None else tf.float32
        return tf.zeros([batch_size, max_time], dtype=dtype)

    def get_initial_attention(self, batch_size, dtype=None):
        dtype = dtype if dtype is not None else tf.float32
        return tf.zeros([batch_size, self.attention_dim], dtype=dtype)

    def call(self, query, memory, prev_alignments,
             prev_max_attentions, memory_lengths=None):
        """
        :param query: [batch, query_depth]
        :param memory: [batch, max_time, memory_depth]
        :param prev_alignments: [batch, max_time], previous alignments
        :param prev_max_attentions: [batch, ], argmax(state, axis=1)
        :param memory_lengths: [batch, memories' lengths]
        :return: alignments: [batch, max_time],
                 next_state: [batch, max_time],
                 max_attentions: [batch, ]
        """
        processed_query = self.query_layer(query) if self.query_layer is not None else query
        values = self.memory_layer(memory) if self.memory_layer is not None else memory
        # [batch, max_time] -> [batch, max_time, 1]
        expanded_alignments = tf.expand_dims(prev_alignments, axis=2)
        # [batch, max_time, attention_filter]
        f = self.location_convolution(expanded_alignments)
        # [batch, max_time, attention_dim]
        processed_location_features = self.location_layer(f)
        # energy: [batch, max_time]
        energy = self._location_sensitive_score(
            processed_query, processed_location_features, values)

        # apply mask
        length_mask = tf.sequence_mask(memory_lengths, name='length_mask') \
            if memory_lengths is not None else None
        max_time = tf.shape(energy)[-1]
        if self.constraint_type is 'monotonic':
            mask_lengths = tf.math.maximum(
                0, max_time - self.window_size - prev_max_attentions)
            key_mask = tf.sequence_mask(prev_max_attentions, max_time)
            reverse_mask = tf.sequence_mask(mask_lengths, max_time)[:, ::-1]
        elif self.constraint_type is 'window':
            key_mask_lengths = tf.math.maximum(
                prev_max_attentions - (self.window_size // 2 +
                                       (self.window_size % 2 != 0)), 0)
            reverse_mask_lens = tf.math.maximum(
                0, max_time - (self.window_size // 2) - prev_max_attentions)
            key_mask = tf.sequence_mask(key_mask_lengths, max_time)
            reverse_mask = tf.sequence_mask(reverse_mask_lens, max_time)[:, ::-1]
        else:
            assert self.constraint_type is None
            key_mask = tf.dtypes.cast(tf.zeros_like(energy), tf.bool)
            reverse_mask = tf.dtypes.cast(tf.zeros_like(energy), tf.bool)
        mask = tf.math.logical_and(
            tf.math.logical_not(
                tf.math.logical_or(key_mask, reverse_mask)),
            length_mask)
        paddings = tf.ones_like(energy) * (-2. ** 32 + 1)
        energy = tf.where(mask, energy, paddings)
        # alignments shape = energy shape = [batch, max_time]
        alignments = tf.math.softmax(energy, axis=1)
        max_attentions = tf.math.argmax(alignments, axis=[1])

        # compute context vector
        # [batch, max_time] -> [batch, 1, max_time]
        expanded = tf.expand_dims(alignments, axis=[1])
        # context: [batch, 1, attention_dim]
        attention = tf.linalg.matmul(expanded, values)
        attention = tf.squeeze(attention, axis=[1])

        # cumulative alignments
        next_state = alignments + prev_alignments if self._cumulate else alignments

        return attention, alignments, next_state, max_attentions


class PreNet(tf.keras.layers.Layer):
    def __init__(self, units, drop_rate, is_training,
                 activation, name='PreNet', **kwargs):
        super(PreNet, self).__init__(name=name, **kwargs)
        self.training = is_training
        self.dense1 = tf.keras.layers.Dense(
            units=units, activation=activation, name='dense_1')
        self.dense2 = tf.keras.layers.Dense(
            units=units, activation=activation, name='dense_2')
        self.dropout_layer = tf.keras.layers.Dropout(rate=drop_rate)

    def call(self, x):
        dense1_out = self.dense1(x)
        dense1_out = self.dropout_layer(dense1_out, training=self.training)
        dense2_out = self.dense2(dense1_out)
        dense2_out = self.dropout_layer(dense2_out)
        return dense2_out


class PostNet(tf.keras.layers.Layer):
    def __init__(self, n_conv, conv_filters, conv_kernel,
                 drop_rate, is_training, name='PostNet', **kwargs):
        super(PostNet, self).__init__(name=name, **kwargs)
        self.conv_stack = []
        self.batch_norm_stack = []
        for i in range(n_conv):
            conv = tf.keras.layers.Conv1D(
                filters=conv_filters, kernel_size=conv_kernel,
                padding='same', activation=None, name='conv_{}'.format(i))
            self.conv_stack.append(conv)
            bn = tf.keras.layers.BatchNormalization(name='batch_norm_{}'.format(i))
            self.batch_norm_stack.append(bn)
        self.dropout_layer = tf.keras.layers.Dropout(rate=drop_rate,
                                                     name='Postnet_dropout')
        self.training = is_training

    def call(self, inputs):
        conv_out = inputs
        activations = [tf.math.tanh] * (len(self.conv_stack) - 1) + [tf.identity]
        for conv, bn, act in zip(self.conv_stack,
                                 self.batch_norm_stack,
                                 activations):
            conv_out = conv(conv_out)
            conv_out = bn(conv_out, training=self.training)
            conv_out = act(conv_out)
            conv_out = self.dropout_layer(conv_out, training=self.training)
        return conv_out


class TacotronDecoderCellState(collections.namedtuple(
    'TacotronDecoderCellState',
    ('cell_state', 'attention', 'time', 'alignments',
     'alignment_history', 'max_attentions'))):
    """
    nametuple storing the state of a TacotronDecoderCell.
    cell_state: The state of the Wrapped RNNCell at the previous time step.
    attention: The attention emitted at the previous time step.
    time: int32 scalar containing the current time step
    alignments: A single or tuple of `TensorArray`(s) containing alignments
                matrices from all time steps.
    """

    def replace(self, **kwargs):
        """
        Clones the current state while overwriting components provided by kwargs
        """
        return super(TacotronDecoderCellState, self)._replace(**kwargs)


TacotronDecoderInput = collections.namedtuple(
    'TacotronDecoderInput', ['decoder_input', 'encoder_output', 'memory_lengths'])


class TacotronDecoderCell(tf.keras.layers.AbstractRNNCell):

    def __init__(self, prenet_units, prenet_rate, n_lstm, lstm_units,
                 lstm_rate, attention_dim, attention_window_size,
                 attention_filters, attention_kernel, cumulate_attention,
                 attention_constraint_type, out_units,
                 is_training, **kwargs):
        self.prenet_units = prenet_units
        self.prenet_rate = prenet_rate
        self.n_lstm = n_lstm
        self.lstm_units = lstm_units
        self.lstm_rate = lstm_rate
        self.attention_dim = attention_dim
        self.attention_filters = attention_filters
        self.attention_kernel = attention_kernel
        self.attention_window_size = attention_window_size
        self.accumulate_attention = cumulate_attention
        self.attention_constraint = attention_constraint_type
        self.out_units = out_units
        self.training = is_training
        super(TacotronDecoderCell, self).__init__(**kwargs)

    def build(self, input_shapes):
        # input_shapes: [(batch, dim), (batch, max_time, encoder_dim)]
        # alignment size is the max_time
        self.batch_size = input_shapes.decoder_input[0]
        self.alignment_size = input_shapes.encoder_output[1]
        self.prenet_layer = PreNet(units=self.prenet_units,
                                   drop_rate=self.prenet_rate,
                                   is_training=self.training,
                                   activation=tf.math.relu)
        lstm_stack = []
        for i in range(self.n_lstm):
            lstm = tf.keras.layers.LSTMCell(units=self.lstm_units,
                                            dropout=self.lstm_rate,
                                            name='decoder_lstm_cell_{}'.format(i))
            lstm_stack.append(lstm)
        self.stacked_lstm_cells = tf.keras.layers.StackedRNNCells(
            lstm_stack, name='decoder_lstm_stacked_cells')
        self.attention_layer = AttentionLayer(
            attention_dim=self.attention_dim,
            attention_filters=self.attention_filters,
            attention_kernel=self.attention_kernel,
            attention_window_size=self.attention_window_size,
            cumulate_weight=self.accumulate_attention,
            constraint_type=self.attention_constraint)
        self.frame_projection = tf.keras.layers.Dense(units=self.out_units,
                                                      activation=None,
                                                      name='frame_projection')
        self.stop_projection = tf.keras.layers.Dense(units=1,
                                                     activation=tf.math.sigmoid,
                                                     name='stop_projection')
        self.built = True

    @property
    def output_size(self):
        return [self.out_units, ], [1, ]

    @property
    def state_size(self):
        return TacotronDecoderCellState(
            cell_state=self.stacked_lstm_cells.state_size,
            attention=self.attention_dim,
            time=tf.TensorShape([]),
            alignments=self.alignment_size,
            alignment_history=(),
            max_attentions=())

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        initial_lstm_cell_states = self.stacked_lstm_cells.get_initial_state(
            None, batch_size, dtype)
        initial_alignments = self.attention_layer.get_initial_alignments(
            self.batch_size, self.alignment_size, dtype=dtype)
        initial_attention = self.attention_layer.get_initial_attention(
            self.batch_size, dtype)
        initial_alignment_history = tf.TensorArray(dtype, size=0, dynamic_size=True)
        initial_max_attentions = tf.zeros((self.batch_size, ), dtype=tf.int32)
        return TacotronDecoderCellState(
            cell_state=initial_lstm_cell_states,
            time=tf.zeros([], dtype=tf.int32),
            attention=initial_attention,
            alignments=initial_alignments,
            alignment_history=initial_alignment_history,
            max_attentions=initial_max_attentions)

    def call(self, inputs, states):
        """
        :param inputs: [batch, max_time, dim]
        :param states:
        :return:
        """
        decoder_input, encoder_output, memory_lengths = tf.nest.flatten(inputs)
        prenet_out = self.prenet_layer(decoder_input)
        lstm_input = tf.concat([prenet_out, states.attention])
        lstm_output, next_cell_state = self.stacked_lstm_cells(
            lstm_input, states.cell_state)
        prev_alignments = states.alignments
        prev_alignment_history = states.alignment_history
        contexts, alignments, cumulated_alignments, max_attentions = self.attention_layer(
            query=lstm_output,
            memory=encoder_output,
            prev_alignments=prev_alignments,
            prev_max_attentions=states.max_attentions,
            memory_lengths=memory_lengths)
        projection_inputs = tf.concat([lstm_output, contexts], axis=-1)
        # compute predicted frames and stop tokens
        cell_outputs = self.frame_projection(projection_inputs)
        stop_tokens = self.stop_projection(projection_inputs)

        # save alignment history
        alignment_history = prev_alignment_history.write(states.time, alignments)
        new_states = TacotronDecoderCellState(
            time=states.time + 1,
            cell_state=next_cell_state,
            attention=contexts,
            alignments=cumulated_alignments,
            alignment_history=alignment_history,
            max_attentions=max_attentions)
        return (cell_outputs, stop_tokens), new_states
