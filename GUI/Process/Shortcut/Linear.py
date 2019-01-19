import tensorflow as tf;
from HNet2_Core import HNet;
from HNet_Enum import Tensor_Type;
from GUI.GUI_Enum import Func_Type, Parameter_Type;
import Compatible_Functions;
from collections import OrderedDict;

description = '''
Construct a multi layer perceptron model with linearly hidden stacks.
1) input_units: Sets the size of input vector.
2) hidden_units: Sets the size of each hidden vector. You can use ',' to construct multiple layers.
3) hidden_activation_func: Sets the activation function of the hidden layer.
4) output units: Sets the size of the output vector.
5) output_activation_func: Set activation function of hidden layer.
6) learning rate: Sets the learning rate.
'''

parameter_Dict = OrderedDict([
    ('input_units', Parameter_Type.Positive_int),
    ('hidden_units', Parameter_Type.Positive_int_list),
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
        shape= [parameters['input_units']]
        )
    hNet.process_Manager.Tensor_Generate(        
        process_Name= parameters['process_Name'],
        name= 'Hidden_0',
        input_Tensor_Names= 'Input',
        tensor_Func= tf.layers.dense,
        reuse_Tensor_Name= None,
        units= parameters['hidden_units'][0],        
        use_bias= True
        )
    hNet.process_Manager.Tensor_Generate(        
        process_Name= parameters['process_Name'],
        name= 'Hidden_0_Act',
        input_Tensor_Names= 'Hidden_0',
        tensor_Func= parameters['hidden_activation_func']
        )

    for index, units in enumerate(parameters['hidden_units'][1:], 1):
        hNet.process_Manager.Tensor_Generate(        
            process_Name= parameters['process_Name'],
            name= 'Hidden_{}'.format(index),
            input_Tensor_Names= 'Hidden_{}'.format(index - 1),
            tensor_Func= tf.layers.dense,
            reuse_Tensor_Name= None,
            units= units,
            use_bias= True
            )
        hNet.process_Manager.Tensor_Generate(        
            process_Name= parameters['process_Name'],
            name= 'Hidden_{}_Act'.format(index),
            input_Tensor_Names= 'Hidden_{}'.format(index),
            tensor_Func= parameters['hidden_activation_func']
            )

    hNet.process_Manager.Tensor_Generate(
        process_Name= parameters['process_Name'],
        name= 'Output',
        input_Tensor_Names= 'Hidden_{}_Act'.format(len(parameters['hidden_units']) - 1),
        tensor_Func= tf.layers.dense,
        reuse_Tensor_Name= None,
        units= parameters['output_units'],
        use_bias= True
        )    
    hNet.process_Manager.Placeholder_Generate(
        process_Name= parameters['process_Name'],
        name= 'Target',
        dtype= tf.float32,
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