import tensorflow as tf
import tensorflow.contrib.slim as slim


class ConvRNNCell(object):
    def __call__(self, inputs, state, scope=None):
        raise NotImplementedError("Abstract method")

    @property
    def state_size(self):
        raise NotImplementedError("Abstract method")

    @property
    def output_size(self):
        raise NotImplementedError("Abstract method")

    def zero_state(self, batch_size, dtype):
        shape = self.shape
        num_features = self.num_features
        zeros = tf.zeros([batch_size, shape[0], shape[1], num_features * 2])
        return zeros


class BasicConvLSTMCell(ConvRNNCell):
    def __init__(self, shape, filter_size, num_features, forget_bias=1.0, input_size=None,
                 state_is_tuple=False, activation=tf.nn.tanh):
        if input_size is not None:
            logging.warn("%s: The input_size parameter is deprecated.", self)
        self.shape = shape
        self.filter_size = filter_size
        self.num_features = num_features
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._activation = activation

    @property
    def state_size(self):
        return (LSTMStateTuple(self._num_units, self._num_units)
                if self._state_is_tuple else 2 * self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope='convLSTM'):
        """Long short-term memory cell (LSTM)."""
        with tf.variable_scope(scope or type(self).__name__):  # "BasicLSTMCell"
            # Parameters of gates are concatenated into one multiply for efficiency.
            if self._state_is_tuple:
                c, h = state
            else:
                c, h = tf.split(state, 2, 3)
            concat = _conv_linear([inputs, h], self.filter_size, self.num_features * 4, True)

            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            i, j, f, o = tf.split(concat, 4, 3)

            new_c = (c * tf.nn.sigmoid(f + self._forget_bias) + tf.nn.sigmoid(i) *
                     self._activation(j))
            new_h = self._activation(new_c) * tf.nn.sigmoid(o)

            if self._state_is_tuple:
                new_state = LSTMStateTuple(new_c, new_h)
            else:
                new_state = tf.concat([new_c, new_h], 3)
            return new_h, new_state


def _conv_linear(args, filter_size, num_features, bias, bias_start=0.0, scope=None):
    dtype = [a.dtype for a in args][0]

    with slim.arg_scope([slim.conv2d], stride=1, padding='SAME', activation_fn=None, scope=scope,
                        weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0e-3),
                        biases_initializer=bias and tf.constant_initializer(bias_start, dtype=dtype)):
        if len(args) == 1:
            res = slim.conv2d(args[0], num_features, [filter_size[0], filter_size[1]], scope='LSTM_conv')
        else:
            res = slim.conv2d(tf.concat(args, 3), num_features, [filter_size[0], filter_size[1]], scope='LSTM_conv')
        return res