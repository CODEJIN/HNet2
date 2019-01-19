import tensorflow as tf;
from tensorflow.contrib.rnn import RNNCell;

class FeedbackCell(RNNCell):
    '''
    [I, H, P] x W = [H, P]
    state: concat([act(hidden), act(projection)]).
    rnn_output: concat([act(hidden), projection]). The activation function is not applied to projection.
    state and output size: 'num_hidden_units + num_projection_units'.
    I recommend to split the rnn_output by (num_hidden_units, num_projection_units).
    '''
    def __init__(
        self,
        num_hidden_units,
        num_projection_units,
        initializer= None,
        use_bias = True,
        bias_initializer = tf.zeros_initializer,
        hidden_activation=None, #Baisc is tanh.        
        projection_state_activation=None, #This is only used for state.
        reuse=None,
        name=None
        ):
        super(FeedbackCell, self).__init__(_reuse=reuse, name=name)
        self._num_hidden_units = num_hidden_units;        
        self._num_projection_units = num_projection_units;        
        self._initializer= initializer;        
        self._use_bias = use_bias;
        self._bias_initializer = bias_initializer;
        self._hidden_activation = hidden_activation or tf.nn.tanh;
        self._projection_state_activation = projection_state_activation or tf.nn.softmax;
        self._reuse = reuse;
        self._name = name;
               
        self._state_size = num_hidden_units + num_projection_units;        
        self._output_size = num_hidden_units + num_projection_units;

    @property
    def state_size(self):
        return self._state_size;
    
    @property
    def output_size(self):
        return self._output_size;

    def call(self, inputs, state):
        input_Size = inputs.get_shape().with_rank(2)[1];

        with tf.variable_scope(self._name or type(self).__name__):
            if input_Size.value is None:
                raise ValueError("Could not infer input size from inputs.get_shape()[-1]")

            kernel = tf.layers.dense(
                inputs= tf.concat([inputs, state], axis = -1),
                units= self._num_hidden_units + self._num_projection_units,
                use_bias= False,
                name= "kernel"
                )
            hidden, projection = tf.split(kernel, num_or_size_splits=[self._num_hidden_units, self._num_projection_units], axis=-1);
            hidden_Act = self._hidden_activation(hidden);
            projection_Act = self._projection_state_activation(projection);
            new_State = tf.concat([hidden_Act, projection_Act], axis=-1);

        return tf.concat([hidden_Act, projection], axis=-1), new_State