#Embedding
import tensorflow as tf;
import inspect;

def embedding(inputs, id_size, embedding_size, name= None, reuse=None, **kwargs):
    ban_parameter_list = ['shape', 'trainable', 'params', 'ids']
    
    embedding_Parameters = {};
    lookup_Parameters = {};
    for parameter, value in kwargs.items():
        if parameter in [x for x in inspect.getargspec(tf.get_variable).args if not x in ban_parameter_list]:
            embedding_Parameters[parameter] = value;
        elif parameter in [x for x in inspect.getargspec(tf.nn.embedding_lookup).args if not x in ban_parameter_list]:
            lookup_Parameters[parameter] = value;
        else:
            raise TypeError("embedding() got an unexpected keyword argument '{}'".format(parameter));
        
    name = name or 'embedding';
    
    with tf.variable_scope(name, reuse=reuse):
        embedding_v = tf.get_variable(
            name = 'embedding_v',
            shape = (id_size, embedding_size),
            dtype = tf.float32,
            trainable = False,
            **embedding_Parameters
            )
    
        lookup = tf.nn.embedding_lookup(
            params= embedding_v,
            ids= inputs,
            name='lookup',
            **lookup_Parameters
            )

    return tf.identity(lookup, name=name)

def add_noise_normal(inputs, name= None, **kwargs):
    kwargs = {key: value for key, value in kwargs.items() if key != "shape"};
    return tf.identity(inputs + tf.random_normal(shape= tf.shape(inputs), **kwargs), name= name or 'add_noise_normal');

def add_noise_uniform(inputs, name= None, **kwargs):
    kwargs = {key: value for key, value in kwargs.items() if key != "shape"};
    return tf.identity(inputs + tf.random_uniform(shape= tf.shape(inputs), **kwargs), name= name or 'add_noise_normal');

def tile(input, multiples, name=None):
    return tf.tile(input, multiples=[1] + multiples, name= name);

def reshape(input, shape, name=None):
    return tf.reshape(input, shape=[-1] + shape, name= name);

def indexing(inputs, axis, index, name= None):
    begin_List = [];
    size_List = [];
    for axis_Index, size in  enumerate(inputs.get_shape().as_list()):
        if axis_Index == axis:
            if not size is None and index >= size:
                raise IndexError("index {} is out of bounds for axis {} with size {}".format(index, axis, size))
            begin_List.append(index);
            size_List.append(1);
        else:
            begin_List.append(0);
            size_List.append(-1);
                
    return tf.squeeze(tf.slice(inputs, begin= begin_List, size= size_List), axis=axis, name=name)
    

def mean_squared_error(inputs, name= None):
    mse_Calc = tf.reduce_mean(tf.pow(inputs[0] - inputs[1], 2), axis=-1);
    return tf.identity(mse_Calc, name= name or 'mean_squared_error');

def cross_entropy(inputs, name= None):
    '''
    inputs[0]: labels,
    inputs[1]: logits,
    '''
    ce_Calc = -tf.reduce_mean(inputs[0] * tf.log(inputs[1] + 1e-5) + (1- inputs[0]) * tf.log(1 - inputs[1] + 1e-5), axis=-1);
    return tf.identity(ce_Calc, name= name or 'cross_entropy');

def cosine_similarity(inputs, name= None):
    cs_Calc = tf.reduce_sum(inputs[0] * inputs[1], axis=-1) / (tf.sqrt(tf.reduce_sum(tf.pow(inputs[0], 2), axis=-1)) * tf.sqrt(tf.reduce_sum(tf.pow(inputs[1], 2), axis=-1)));
    return tf.identity(cs_Calc, name= name or 'cosine_similarity');

def euclidean_distance(inputs, name= None):
    ed_Calc = tf.sqrt(tf.reduce_sum(tf.pow(inputs[0] - inputs[1], 2), axis= -1));
    return tf.identity(ed_Calc, name= name or 'euclidean_distance');

def argmax(inputs, name= None):
    return tf.argmax(inputs, axis=-1, name= name or 'euclidean_distance');

def multi_test_calc(inputs, name = None):
    return [
        mean_squared_error(inputs, name='{}_mean_sqared_error'.format(name)),
        cross_entropy(inputs, name='{}_cross_entropy'.format(name)),
        cosine_similarity(inputs, name='{}_cosine_similarity'.format(name)),
        euclidean_distance(inputs, name='{}_euclidean_distance'.format(name))
        ]

def semantic_stress(inputs, name= None):
    def base_log(x, base):        
        return tf.log(x) / tf.log(tf.cast(base, dtype=tf.float32))

    ss_Calc = tf.reduce_mean(inputs * base_log(inputs + 1e-5, 2) + (1 - inputs) * base_log(1 - inputs + 1e-5, 2) + 1, axis= -1)
    return tf.identity(ss_Calc, name= name or 'semantic_stress');