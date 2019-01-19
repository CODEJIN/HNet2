import tensorflow as tf;
from HNet2_Core import HNet;
from HNet_Enum import Tensor_Type;
from GUI.GUI_Enum import Func_Type, Parameter_Type;
import Compatible_Functions;
from collections import OrderedDict;

description = '''
Configure the RNN model.
1) time_step: Set time size of input data.
2) input_units: Sets the size of input vector.
3) hidden_units: Sets the size of hidden vector.
4) hidden_Cell_func: Set the RNN cell to use.
5) hidden_state_reset: If True, the initial state at the beginning of the training is the last state of the previous learning. If False, initial state is used where all values are zero. Test does not apply the last state of previous learning (always 0 state).
6) output units: Sets the size of the output vector.
7) output_activation_func: Set activation function of hidden layer.
8) learning rate: Sets the learning rate.
'''

parameter_Dict = OrderedDict([
    ('time_step', Parameter_Type.Positive_int),
    ('input_units', Parameter_Type.Positive_int),
    ('hidden_units', Parameter_Type.Positive_int),
    ('hidden_Cell_func', Parameter_Type.RNN_cell_func),
    ('hidden_state_reset', Parameter_Type.Bool),
    ('output_units', Parameter_Type.Positive_int),
    ('output_activation_func', Parameter_Type.Output_activation_func),
    ('learning_rate', Parameter_Type.Positive_float),
    ])

def Run_Shortcut(hNet, **parameters):
    hNet.process_Manager.Placeholder_Generate(
        process_Name= parameters['process_Name'],
        name= 'Input',
        dtype= tf.float32,
        shape= [parameters['time_step'], parameters['input_units']]
        )
    hNet.process_Manager.RNN_Tensor_Generate(
        process_Name= parameters['process_Name'],
        name= 'RNN_Hidden',
        input_Tensor_Name= 'Input',
        num_units= parameters['hidden_units'],
        rnn_Cell_Func = parameters['hidden_Cell_func'][0],
        state_Func = parameters['hidden_Cell_func'][1],
        state_Reset = parameters['hidden_state_reset']
        )
    hNet.process_Manager.Tensor_Generate(        
        process_Name= parameters['process_Name'],
        name= 'Output',
        input_Tensor_Names= 'RNN_Hidden',
        tensor_Func= tf.layers.dense,
        reuse_Tensor_Name= None,
        units= parameters['output_units'],        
        use_bias= True
        )      
    hNet.process_Manager.Placeholder_Generate(
        process_Name= parameters['process_Name'],
        name= 'Target',
        dtype= tf.float32,
        shape= [parameters['time_step'], parameters['output_units']]
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