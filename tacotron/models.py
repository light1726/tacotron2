import tensorflow as tf
from tacotron.modules import TacotronEncoder, TacotronDecoderCell, PostNet, TacotronDecoderInput


class Tacotron(tf.keras.Model):
    def __init__(self, input_dim, enc_emb_size, enc_nconv, enc_conv_kernel,
                 enc_conv_filter, enc_lstm_hidden, att_dim, att_win, att_filter,
                 att_kernel, att_cumulate_weight, att_constraint, pre_units,
                 pre_rate, dec_nlstm, dec_lstm_units, dec_lstm_rate, dec_out_dim,
                 post_nconv, post_conv_filter, post_conv_kernel, post_rate,
                 is_training, name='Tacotron', **kwargs):
        super(Tacotron, self).__init__(name=name, **kwargs)
        self.encoder = TacotronEncoder(input_dim=input_dim, emb_size=enc_emb_size,
                                       n_conv=enc_nconv, conv_kernel=enc_conv_kernel,
                                       conv_filter=enc_conv_filter,
                                       lstm_hidden=enc_lstm_hidden,
                                       is_training=is_training)
        self.decoder_cell = TacotronDecoderCell(prenet_units=pre_units,
                                                prenet_rate=pre_rate,
                                                n_lstm=dec_nlstm,
                                                lstm_units=dec_lstm_units,
                                                lstm_rate=dec_lstm_rate,
                                                attention_dim=att_dim,
                                                attention_window_size=att_win,
                                                attention_filters=att_filter,
                                                attention_kernel=att_kernel,
                                                cumulate_attention=att_cumulate_weight,
                                                attention_constraint_type=att_constraint,
                                                out_units=dec_out_dim,
                                                is_training=is_training)
        self.postnet = PostNet(n_conv=post_nconv, conv_filters=post_conv_filter,
                               conv_kernel=post_conv_kernel, drop_rate=post_rate,
                               is_training=is_training)
        self.post_projection = tf.keras.layers.Dense(units=dec_out_dim, activation=None,
                                                     name='residual_projection')

    def call(self, text_inputs, mel_outputs, input_lengths=None, output_lenths=None):
        """
        :param text_inputs: [batch, max_time, in_dim]
        :param mel_outputs: [batch, max_time, mel_dim]
        :param input_lengths: [batch, ]
        :param output_lenths: [batch, ]
        :return:
        """
        max_time = tf.shape(mel_outputs)[1]
        time_first_targets = tf.reshape(mel_outputs, [1, 0, 2])
        encoder_outs = self.encoder(text_inputs)
        frame_predictions = tf.TensorArray(tf.float32, mel_outputs.shape[0])
        stop_predictions = tf.TensorArray(tf.float32, mel_outputs.shape[0])
        state = self.decoder_cell.get_initial_state(
            inputs=None, batch_size=tf.shape(text_inputs)[0], dtype=tf.float32)
        for i in tf.range(max_time):
            decoder_input = TacotronDecoderInput(
                time_first_targets[i], encoder_outs, input_lengths)
            output, state = self.decoder_cell(decoder_input, state)
            frame_pred, stop_pred = output
            frame_predictions = frame_predictions.write(i, frame_pred)
            stop_predictions = stop_predictions.write(i, stop_pred)
        mel_outputs = tf.transpose(frame_predictions.stack(), [1, 0, 2])
        stop_outputs = tf.transpose(stop_predictions.stack(), [1, 0, 2])
        residual = self.post_projection(mel_outputs)
        residual = self.post_projection(residual)
        refined_outputs = mel_outputs + residual

