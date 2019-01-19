import tensorflow as tf;
from HNet2_Core import HNet;
from HNet_Enum import Tensor_Type;
from GUI.GUI_Enum import Func_Type, Parameter_Type;
import Compatible_Functions;
from collections import OrderedDict;

description = '''
Construct a multi-layer model with linearly CNN layers stacks.
1) input_shape: set the shape of input data. The shape must be of the form 'Height, Width, channel'.
2) filters: Decide how many filters to use for the operation. You can use ',' to construct multiple layers. The number of filter should be same to 'kernel_sizes'.
3) kernel_sizes: Determines the size of the filter used in the operation.  You can use ',' to construct multiple layers.  The number of filter should be same to 'filters'.
4) hidden_activation_func: Sets the activation function of the hidden layer.
4) output units: Sets the size of the output vector.
5) output_activation_func: Set activation function of hidden layer.
6) learning rate: Sets the learning rate.
'''

parameter_Dict = OrderedDict([
    ('input_shape', Parameter_Type.Positive_int_list),
    ('filters', Parameter_Type.Positive_int_list),
    ('kernel_sizes', Parameter_Type.Positive_int_list),       
    ('padding', Parameter_Type.Padding),    
    ('hidden_activation_func', Parameter_Type.Hidden_activation_func),
    ('output_units', Parameter_Type.Positive_int),
    ('output_activation_func', Parameter_Type.Output_activation_func),
    ('learning_rate', Parameter_Type.Positive_float),
    ])

def Run_Shortcut(hNet, **parameters):
    hNet.process_Manager.Placeholder_Generate(
        process_Name= parameters['process_Name'],
        name= 'Input',
        dtype= tf.float32,
        shape= parameters['input_shape']
        )
    
    for index, (filters, kernel_size) in enumerate(zip(parameters['filters'], parameters['kernel_sizes'])):
        hNet.process_Manager.Tensor_Generate(
            process_Name= parameters['process_Name'],
            name= 'Hidden_{}'.format(index),
            input_Tensor_Names= 'Hidden_{}_Act'.format(index - 1) if index > 0 else 'Input',
            tensor_Func= tf.layers.conv2d,
            filters= filters,
            kernel_size= (kernel_size, kernel_size),
            strides= (1,1),
            padding= parameters['padding']
            )
        hNet.process_Manager.Tensor_Generate(
            process_Name= parameters['process_Name'],
            name= 'Hidden_{}_Act'.format(index),
            input_Tensor_Names= 'Hidden_{}'.format(index),
            tensor_Func= parameters['hidden_activation_func']
            )
        
    hNet.process_Manager.Tensor_Generate(
        process_Name= parameters['process_Name'],
        name= 'Flatten',
        input_Tensor_Names= 'Hidden_{}_Act'.format(len(parameters['filters']) - 1),
        tensor_Func= tf.layers.flatten
        )
        
    hNet.process_Manager.Tensor_Generate(
        process_Name= parameters['process_Name'],
        name= 'Output',
        input_Tensor_Names= 'Flatten',
        tensor_Func= tf.layers.dense,
        reuse_Tensor_Name= None,
        units= parameters['output_units'],
        use_bias= True
        )

    hNet.process_Manager.Placeholder_Generate(
        process_Name= parameters['process_Name'],
        name= 'Target',
        dtype= tf.int32,
        shape= [parameters['output_units']]
        )

    if parameters['output_activation_func'] == tf.nn.sigmoid:
        hNet.process_Manager.Loss_Generate(
            process_Name= parameters['process_Name'],
            name= 'Loss',
            label_Tensor_Name= 'Target',
            loss_Func= tf.losses.sigmoid_cross_entropy,
            logit_Tensor_Name= 'Output',
            prediction_Tensor_Name= None,
            weights= 1.0
            )
    elif parameters['output_activation_func'] == tf.nn.softmax:
        hNet.process_Manager.Loss_Generate(
            process_Name= parameters['process_Name'],
            name= 'Loss',
            label_Tensor_Name= 'Target',
            loss_Func= tf.losses.softmax_cross_entropy,
            logit_Tensor_Name= 'Output',
            prediction_Tensor_Name= None,
            weights= 1.0
            )
    
    hNet.process_Manager.Optimizer_Generate(
        process_Name= parameters['process_Name'],
        optimizer_Func= tf.train.AdamOptimizer,
        initial_learning_rate= parameters['learning_rate'],        
        )

    hNet.process_Manager.Tensor_Generate(
        process_Name= parameters['process_Name'],
        name= 'Output_Act',
        input_Tensor_Names= 'Output',
        tensor_Func= parameters['output_activation_func']
        )
    hNet.process_Manager.Tensor_Generate(
        process_Name= parameters['process_Name'],
        name= 'Test',
        input_Tensor_Names= ['Target', 'Output_Act'],
        tensor_Func= Compatible_Functions.Custom.multi_test_calc,
        )