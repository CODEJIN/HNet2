import tensorflow as tf;
from HNet2_Core import HNet;
from HNet_Enum import Tensor_Type;
from GUI.GUI_Enum import Func_Type, Parameter_Type;
import Compatible_Functions;
from collections import OrderedDict;

func_Dict = OrderedDict();

#Placeholder
func_Dict["Placeholder"] = {
    "Func_Type": Func_Type.Placeholder,
    "Tensor_Type": Tensor_Type.Placeholder,
    "Parameter": OrderedDict([
        ("name", Parameter_Type.Name),
        ("dtype", Parameter_Type.Dtype),
        ("shape", Parameter_Type.Positive_int_list)
        ]),
    "Required": ["name", "dtype"],
    "Description": '''
Add a placeholder within the process. Plaecholder accepts three parameters: name, dtype, shape. Name and dtype parameters are required.
1) Name: An identifier within the process. This value does not affect learning.
2) Dtype: Determines the type of placeholder. You can enter one of three types: float32, int32, bool.
3) shape: Determines the shape of the placeholder. This parameter can contain a single number or a list of numbers separated by ','. Please make sure that the pattern matches the pattern you want to insert in the future. If no value is assigned, it is assumed that scalar will be entered. ※Unstructured type currently does not support user's setting.
'''
    }

#Forward
func_Dict["Layer_Dense"] = {
    "Func_Type": Func_Type.Forward,
    "Tensor_Type": Tensor_Type.Tensor,
    "Parameter": OrderedDict([
        ("name", Parameter_Type.Name),
        ("input_Tensor_Names", Parameter_Type.Tensor),
        ("tensor_Func", tf.layers.dense),
        ("units", Parameter_Type.Positive_int),
        ("use_bias", Parameter_Type.Bool),
        ("reuse_Tensor_Name", Parameter_Type.Process_and_Tensor_with_None)
        ]),
    "Required": ["name", "input_Tensor_Names", "tensor_Func", "units"],
    "Description": '''
Layer dense adds a one-dimensional vector tensor of the specified size by matrix multiplying the input one-dimensional vector and a weight matrix. If the input tensor is n-dimensional, the last n-dimensional values are treated as a vector and the output is also an n-dimensional tensor with the last dimension specified (e.g. [B,3,5,7] -dense-> [B,3,5, 10]). The parameter list is as follows.
1) Name: An identifier within the process. This value does not affect learning.
2) input_Tensor_Name: Determines from which tensor the calculation is to proceed. This value is required.
3) units: Determines the size of the output vector. This value is required.
4) use_bias: Decides whether to add a bias.
5) reuse_Tensor_Name: Import weight from another layer dense. In this case, the weight is shared with another layer.
'''
    }

func_Dict["Conv1d"] = {
    "Func_Type": Func_Type.Forward,
    "Tensor_Type": Tensor_Type.Tensor,
    "Parameter": OrderedDict([
        ("name", Parameter_Type.Name),
        ("input_Tensor_Names", Parameter_Type.Tensor),
        ("tensor_Func", tf.layers.conv1d),
        ("filters", Parameter_Type.Positive_int),
        ("kernel_size", Parameter_Type.Positive_int),
        ("strides", Parameter_Type.Positive_int),
        ("padding", Parameter_Type.Padding),
        ("use_bias", Parameter_Type.Bool),
        ("reuse_Tensor_Name", Parameter_Type.Process_and_Tensor_with_None)
        ]),
    "Required": ["name", "input_Tensor_Names", "tensor_Func", "filters", "kernel_size"],
    "Description": '''
Apply filters to input 3d data and convolute to other 3d data.
1) Name: An identifier within the process. This value does not affect learning.
2) input_Tensor_Names: Determines from which tensor the calculation is to proceed. The shape of the input tensor must be 3D in the form of [B, step, channel]. This value is required.
3) filters: Decide how many filters to use for the operation. This value is the same as the channel number of the tensor to be output. This value is required.
4) kernel_size: Determines the size of the filter used in the operation. This value is required.
5) strides: Determines the step size of filter movement at the time of convolution.
6) padding: Determines whether the step size after convolution is equal to the entered tensor by padding. 
7) use_bias: Decides whether to add a bias.
5) reuse_Tensor_Name: Import weight from another conv1d. In this case, the weight is shared with another layer.
'''
    }

func_Dict["Conv2d"] = {
    "Func_Type": Func_Type.Forward,
    "Tensor_Type": Tensor_Type.Tensor,
    "Parameter": OrderedDict([
        ("name", Parameter_Type.Name),
        ("input_Tensor_Names", Parameter_Type.Tensor),
        ("tensor_Func", tf.layers.conv2d),
        ("filters", Parameter_Type.Positive_int),
        ("kernel_size", Parameter_Type.Positive_int_list),
        ("strides", Parameter_Type.Positive_int_list),
        ("padding", Parameter_Type.Padding),
        ("use_bias", Parameter_Type.Bool),
        ("reuse_Tensor_Name", Parameter_Type.Process_and_Tensor_with_None)
        ]),
    "Required": ["name", "input_Tensor_Names", "tensor_Func", "filters", "kernel_size"],
    "Description": '''
Apply the filter to the input 4d [B, height, width, channel] data and convolute it to other 4d data.
1) Name: An identifier within the process. This value does not affect learning.
2) input_Tensor_Names: Determines from which tensor the calculation is to proceed. The shape of the input tensor must be 3D in the form of [B, height, width, channel]. This value is required.
3) filters: Decide how many filters to use for the operation. This value is the same as the channel number of the tensor to be output. This value is required.
4) kernel_size: Determines the size of the filter used in the operation. This value is required.
5) strides: Determines the step size of filter movement at the time of convolution.
6) padding: Determines whether the step size after convolution is equal to the entered tensor by padding. 
7) use_bias: Decides whether to add a bias.
5) reuse_Tensor_Name: Import weight from another conv2d. In this case, the weight is shared with another layer.
'''
    }

func_Dict["Conv2d_Transpose"] = {
    "Func_Type": Func_Type.Forward,
    "Tensor_Type": Tensor_Type.Tensor,
    "Parameter": OrderedDict([
        ("name", Parameter_Type.Name),
        ("input_Tensor_Names", Parameter_Type.Tensor),
        ("tensor_Func", tf.layers.conv2d_transpose),
        ("filters", Parameter_Type.Positive_int),
        ("kernel_size", Parameter_Type.Positive_int_list),
        ("strides", Parameter_Type.Positive_int_list),
        ("padding", Parameter_Type.Padding),
        ("use_bias", Parameter_Type.Bool),
        ("reuse_Tensor_Name", Parameter_Type.Process_and_Tensor_with_None)
        ]),
    "Required": ["name", "input_Tensor_Names", "tensor_Func", "filters", "kernel_size"],
    "Description": '''
Apply the filter to the input 4d [B, height, width, channel] data and transposed convolute it to other 4d data.
1) Name: An identifier within the process. This value does not affect learning.
2) input_Tensor_Names: Determines from which tensor the calculation is to proceed. The shape of the input tensor must be 3D in the form of [B, height, width, channel]. This value is required.
3) filters: Decide how many filters to use for the operation. This value is the same as the channel number of the tensor to be output. This value is required.
4) kernel_size: Determines the size of the filter used in the operation. This value is required.
5) strides: Determines the step size of filter movement at the time of convolution.
6) padding: Determines whether the step size after convolution is equal to the entered tensor by padding. 
7) use_bias: Decides whether to add a bias.
5) reuse_Tensor_Name: Import weight from another conv2d transpose. In this case, the weight is shared with another layer.
'''
    }

#RNN
func_Dict["RNN_LSTM"] = {
    "Func_Type": Func_Type.RNN,
    "Tensor_Type": Tensor_Type.RNN_Tensor,
    "Parameter": OrderedDict([
        ("name", Parameter_Type.Name),
        ("input_Tensor_Name", Parameter_Type.Tensor),
        ("rnn_Cell_Func", tf.nn.rnn_cell.LSTMCell),
        ("state_Func", tf.nn.rnn_cell.LSTMStateTuple),
        ("state_Reset", Parameter_Type.Bool),
        ("num_units", Parameter_Type.Positive_int),
        ("reuse_Tensor_Name", Parameter_Type.Process_and_Tensor_with_None)
        ]),
    "Required": ["name", "input_Tensor_Name", "rnn_Cell_Func", "num_units"],
    "Description": '''
Perform RNN operation using LSTM cell.
1) Name: An identifier within the process. This value does not affect learning.
2) input_Tensor_Name: Determines from which tensor the calculation is to proceed. The shape of the input tensor must be 3D in the form [B, time, vector]. This value is required.
3) state_Reset: If True, the initial state at the beginning of the training is the last state of the previous learning. If False, initial state is used where all values are zero. Test does not apply the last state of previous learning (always 0 state).
4) num_units: Determines the size of the output vector. This value is required.
5) reuse_Tensor_Name: Import weight from another RNN LSTM. In this case, the weight is shared with another layer.
'''
    }

func_Dict["RNN_GRU"] = {
    "Func_Type": Func_Type.RNN,
    "Tensor_Type": Tensor_Type.RNN_Tensor,
    "Parameter": OrderedDict([
        ("name", Parameter_Type.Name),
        ("input_Tensor_Name", Parameter_Type.Tensor),
        ("rnn_Cell_Func", tf.nn.rnn_cell.GRUCell),
        ("state_Reset", Parameter_Type.Bool),
        ("num_units", Parameter_Type.Positive_int),        
        ("reuse_Tensor_Name", Parameter_Type.Process_and_Tensor_with_None)
        ]),
    "Required": ["name", "input_Tensor_Name", "rnn_Cell_Func", "num_units"],
    "Description": '''
Perform RNN operation using GRU cell.
1) Name: An identifier within the process. This value does not affect learning.
2) input_Tensor_Name: Determines from which tensor the calculation is to proceed. The shape of the input tensor must be 3D in the form [B, time, vector]. This value is required.
3) state_Reset: If True, the initial state at the beginning of the training is the last state of the previous learning. If False, initial state is used where all values are zero. Test does not apply the last state of previous learning (always 0 state).
4) num_units: Determines the size of the output vector. This value is required.
5) reuse_Tensor_Name: Import weight from another RNN GRU. In this case, the weight is shared with another layer.
'''
    }

func_Dict["RNN_Basic"] = {
    "Func_Type": Func_Type.RNN,
    "Tensor_Type": Tensor_Type.RNN_Tensor,
    "Parameter": OrderedDict([
        ("name", Parameter_Type.Name),
        ("input_Tensor_Name", Parameter_Type.Tensor),
        ("rnn_Cell_Func", tf.nn.rnn_cell.BasicRNNCell),
        ("state_Reset", Parameter_Type.Bool),
        ("num_units", Parameter_Type.Positive_int),
        ("reuse_Tensor_Name", Parameter_Type.Process_and_Tensor_with_None)
        ]),
    "Required": ["name", "input_Tensor_Name", "rnn_Cell_Func", "num_units"],
    "Description": '''
Perform RNN operation using Basic cell (back propagation through time).
1) Name: An identifier within the process. This value does not affect learning.
2) input_Tensor_Name: Determines from which tensor the calculation is to proceed. The shape of the input tensor must be 3D in the form [B, time, vector]. This value is required.
3) state_Reset: If True, the initial state at the beginning of the training is the last state of the previous learning. If False, initial state is used where all values are zero. Test does not apply the last state of previous learning (always 0 state).
4) num_units: Determines the size of the output vector. This value is required.
5) reuse_Tensor_Name: Import weight from another RNN Basic. In this case, the weight is shared with another layer.
'''
    }

func_Dict["RNN_Feedback"] = {
    "Func_Type": Func_Type.RNN,
    "Tensor_Type": Tensor_Type.RNN_Tensor,
    "Parameter": OrderedDict([
        ("name", Parameter_Type.Name),
        ("input_Tensor_Name", Parameter_Type.Tensor),
        ("rnn_Cell_Func", Compatible_Functions.Cells.FeedbackCell),
        ("state_Reset", Parameter_Type.Bool),
        ("num_hidden_units", Parameter_Type.Positive_int),
        ("num_projection_units", Parameter_Type.Positive_int),
        ("projection_state_activation", Parameter_Type.Output_activation_func),
        ("reuse_Tensor_Name", Parameter_Type.Process_and_Tensor_with_None)
        ]),
    "Required": ["name", "input_Tensor_Name", "rnn_Cell_Func", "num_hidden_units", "num_projection_units", "output_state_activation"],
    "Description": '''
Performs RNN operation that receives both basic cell and projection as state. (Elman + Jordan network). This cell outputs both hidden and projection. The size of the output vector is num_hidden_units + num_proejction_units. The hidden vector is applied with tanh and the projection vector is non scaled logit. It is recommended to divide the output tensor by split.
1) Name: An identifier within the process. This value does not affect learning.
2) input_Tensor_Name: Determines from which tensor the calculation is to proceed. The shape of the input tensor must be 3D in the form [B, time, vector]. This value is required.
3) state_Reset: If True, the initial state at the beginning of the training is the last state of the previous learning. If False, initial state is used where all values are zero. Test does not apply the last state of previous learning (always 0 state).
4) num_hidden_units: Determines the size of hidden part of output vector. This value is required.
5) num_projection_units: Determines the size of projection part of output vector. This value is required.
6) projection_state_activation: Determines the activation function to apply to the projection within the RNN operation. This value is required. ※ This value does not apply to the projection vector.
7) reuse_Tensor_Name: Import weight from another RNN Feedback. In this case, the weight is shared with another layer.
'''
    }

#Activation func
func_Dict["Sigmoid"] = {
    "Func_Type": Func_Type.Activation_Calc,
    "Tensor_Type": Tensor_Type.Tensor,
    "Parameter": OrderedDict([
        ("name", Parameter_Type.Name),
        ("input_Tensor_Names", Parameter_Type.Tensor),
        ("tensor_Func", tf.nn.sigmoid)
        ]),
    "Required": ["name", "input_Tensor_Names", "tensor_Func"],
    "Description": '''
Apply the expression y = 1 / (1+exp(-x)) to the tensor to convert it to a value within the range of 0 and 1.
1) Name: An identifier within the process. This value does not affect learning.
2) input_Tensor_Name: Determines from which tensor the calculation is to proceed. This value is required.
'''
    }

func_Dict["Softmax"] = {
    "Func_Type": Func_Type.Activation_Calc,
    "Tensor_Type": Tensor_Type.Tensor,
    "Parameter": OrderedDict([
        ("name", Parameter_Type.Name),
        ("input_Tensor_Names", Parameter_Type.Tensor),
        ("tensor_Func", tf.nn.softmax)
        ]),
    "Required": ["name", "input_Tensor_Names", "tensor_Func"],
    "Description": '''
Apply y = exp(x)/sum(exp(logits)) to the tensor to convert the vector to a probability of the total sum of 1. Logits is the entire vector entered. If more than two dimensions, it applies to the last dimension.
1) Name: An identifier within the process. This value does not affect learning.
2) input_Tensor_Name: Determines from which tensor the calculation is to proceed. This value is required.
'''
    }

func_Dict["Tanh"] = {
    "Func_Type": Func_Type.Activation_Calc,
    "Tensor_Type": Tensor_Type.Tensor,
    "Parameter": OrderedDict([
        ("name", Parameter_Type.Name),
        ("input_Tensor_Names", Parameter_Type.Tensor),
        ("tensor_Func", tf.nn.tanh)
        ]),
    "Required": ["name", "input_Tensor_Names", "tensor_Func"],
    "Description": '''
Apply a hyperbolic tangent(tanh) to the tensor to convert it to a value between -1 and 1.
1) Name: An identifier within the process. This value does not affect learning.
2) input_Tensor_Name: Determines from which tensor the calculation is to proceed. This value is required.
'''
    }

func_Dict["ReLU"] = {
    "Func_Type": Func_Type.Activation_Calc,
    "Tensor_Type": Tensor_Type.Tensor,
    "Parameter": OrderedDict([
        ("name", Parameter_Type.Name),
        ("input_Tensor_Names", Parameter_Type.Tensor),
        ("tensor_Func", tf.nn.relu)
        ]),
    "Required": ["name", "input_Tensor_Names", "tensor_Func"],
    "Description": '''
Apply y = max(x,0) to the tensor to convert all values below zero to zero.
1) Name: An identifier within the process. This value does not affect learning.
2) input_Tensor_Name: Determines from which tensor the calculation is to proceed. This value is required.
'''
    }

func_Dict["Leaky_ReLU"] = {
    "Func_Type": Func_Type.Activation_Calc,
    "Tensor_Type": Tensor_Type.Tensor,
    "Parameter": OrderedDict([
        ("name", Parameter_Type.Name),
        ("input_Tensor_Names", Parameter_Type.Tensor),
        ("tensor_Func", tf.nn.leaky_relu),
        ("alpha", Parameter_Type.Positive_float)
        ]),
    "Required": ["name", "input_Tensor_Names", "tensor_Func"],
    "Description": '''
Apply y = max(x,0) + min(0,alpha*x) to the tensor.
1) Name: An identifier within the process. This value does not affect learning.
2) input_Tensor_Name: Determines from which tensor the calculation is to proceed. This value is required.
3) alpha: Sets the argument alpha to be used for the function. If not set, 0.2 is assigned.
'''
    }

func_Dict["Softplus"] = {
    "Func_Type": Func_Type.Activation_Calc,
    "Tensor_Type": Tensor_Type.Tensor,
    "Parameter": OrderedDict([
        ("name", Parameter_Type.Name),
        ("input_Tensor_Names", Parameter_Type.Tensor),
        ("tensor_Func", tf.nn.softplus)
        ]),
    "Required": ["name", "input_Tensor_Names", "tensor_Func"],
    "Description": '''
Apply y = log(exp(x)+1) to the tensor.
1) Name: An identifier within the process. This value does not affect learning.
2) input_Tensor_Name: Determines from which tensor the calculation is to proceed. This value is required.
'''
    }


#Reshape
func_Dict["Concat"] = {
    "Func_Type": Func_Type.Reshape,
    "Tensor_Type": Tensor_Type.Tensor,
    "Parameter": OrderedDict([
        ("name", Parameter_Type.Name),
        ("input_Tensor_Names", Parameter_Type.Tensor_list),
        ("axis", Parameter_Type.Positive_int),
        ("tensor_Func", tf.concat)
        ]),
    "Required": ["name", "input_Tensor_Names", "tensor_Func", "axis"],
    "Description": '''
The input tensors are combined based on the specified dimension. If it is more than 3D, the dimensions of the remaining dimensions except for the specified dimension must be the same (e.g. [B, 5, 32], [B, 5, 17] -> [B, 5, 49]).
1) Name: An identifier within the process. This value does not affect learning.
2) input_Tensor_Name: Determines from which tensors the calculation is to proceed. This value is required.
3) axis: Specifies the dimension to join. This value is required.
'''
    }

func_Dict["Split"] = {
    "Func_Type": Func_Type.Reshape,
    "Tensor_Type": Tensor_Type.Tensor,
    "Parameter": OrderedDict([
        ("name", Parameter_Type.Name),
        ("input_Tensor_Names", Parameter_Type.Tensor),
        ("num_or_size_splits", Parameter_Type.Positive_int_list),
        ("axis", Parameter_Type.Positive_int),
        ("tensor_Func", tf.split)
        ]),
    "Required": ["name", "input_Tensor_Names", "tensor_Func", "num_or_size_splits", "axis"],
    "Description": '''
Divides input tensor by specified dimension. Tensors are created for the count of numbers assigned to num_or_size_splits.
1) Name: An identifier within the process. This value does not affect learning.
2) input_Tensor_Name: Determines from which tensor the calculation is to proceed. This value is required.
3) num_or_size_splits: Determines the size of each divided tensor. The sum of the numbers in the whole must be equal to the size of the dimension before the split. This value is required.
3) axis: Specifies the dimension to split. This value is required.
'''
    }

func_Dict["Expand_Dims"] = {
    "Func_Type": Func_Type.Reshape,
    "Tensor_Type": Tensor_Type.Tensor,
    "Parameter": OrderedDict([
        ("name", Parameter_Type.Name),
        ("input_Tensor_Names", Parameter_Type.Tensor),
        ("axis", Parameter_Type.Positive_int),
        ("tensor_Func", tf.expand_dims)
        ]),
    "Required": ["name", "input_Tensor_Names", "tensor_Func", "axis"],
    "Description": '''
Inserts the dimension specified in the input tensor. (e.g., [B,3,5,7] -expand dim axis2-> [B,3,1,5,7])
1) Name: An identifier within the process. This value does not affect learning.
2) input_Tensor_Name: Determines from which tensor the calculation is to proceed. This value is required.
3) axis: Specifies the dimension to be inserted. This value is required.
'''
    }

func_Dict["Tile"] = {
    "Func_Type": Func_Type.Reshape,
    "Tensor_Type": Tensor_Type.Tensor,
    "Parameter": OrderedDict([
        ("name", Parameter_Type.Name),
        ("input_Tensor_Names", Parameter_Type.Tensor),
        ("multiples", Parameter_Type.Positive_int_list),
        ("tensor_Func", Compatible_Functions.Custom.tile)
        ]),
    "Required": ["name", "input_Tensor_Names", "tensor_Func", "multiples"],
    "Description": '''
Creates a tensor that is repeated / copied the specified number of times for each dimension.
1) Name: An identifier within the process. This value does not affect learning.
2) input_Tensor_Name: Determines from which tensor the calculation is to proceed. This value is required.
3) multiples: Specifies the number of iterations for each dimension. The count of this value must match the number of dimensions of the tensor to be entered. This value is required.
'''
    }

func_Dict["Reshape"] = {
    "Func_Type": Func_Type.Reshape,
    "Tensor_Type": Tensor_Type.Tensor,
    "Parameter": OrderedDict([
        ("name", Parameter_Type.Name),
        ("input_Tensor_Names", Parameter_Type.Tensor),
        ("shape", Parameter_Type.Positive_int_list),
        ("tensor_Func", Compatible_Functions.Custom.reshape)
        ]),
    "Required": ["name", "input_Tensor_Names", "tensor_Func", "shape"],
    "Description": '''
Overrides the shape of the input tensor.
1) Name: An identifier within the process. This value does not affect learning.
2) input_Tensor_Name: Determines from which tensor the calculation is to proceed. This value is required.
3) shape: Specifies the shape to deform. The sum of the total values must be equal to the sum of the dimensions of the total dimensions of the input tensor. This value is required.
'''
    }

func_Dict["Flatten"] = {
    "Func_Type": Func_Type.Reshape,
    "Tensor_Type": Tensor_Type.Tensor,
    "Parameter": OrderedDict([
        ("name", Parameter_Type.Name),
        ("input_Tensor_Names", Parameter_Type.Tensor),
        ("tensor_Func", tf.layers.flatten)
        ]),
    "Required": ["name", "input_Tensor_Names", "tensor_Func"],
    "Description": '''
Removes all dimensions of the input tensor and converts it to a batch of one-dimensional vector. (e.g. [B, 5, 7] -> [B, 35])
1) Name: An identifier within the process. This value does not affect learning.
2) input_Tensor_Name: Determines from which tensor the calculation is to proceed. This value is required.
'''
    }

func_Dict["Squeeze"] = {
    "Func_Type": Func_Type.Reshape,
    "Tensor_Type": Tensor_Type.Tensor,
    "Parameter": OrderedDict([
        ("name", Parameter_Type.Name),
        ("input_Tensor_Names", Parameter_Type.Tensor),
        ("axis", Parameter_Type.Positive_int_list),
        ("tensor_Func", tf.squeeze)
        ]),
    "Required": ["name", "input_Tensor_Names", "tensor_Func", "axis"],
    "Description": '''
Removes the specified dimensions from the input tensor. The size of specified dimension must be 1.
1) Name: An identifier within the process. This value does not affect learning.
2) input_Tensor_Name: Determines from which tensor the calculation is to proceed. This value is required.
3) axis: Specifies the dimensions to remove. Specified dimensions must be 1 in size. This value is required.
'''
    }

func_Dict["Add_n"] = {
    "Func_Type": Func_Type.Fundamental_Math,
    "Tensor_Type": Tensor_Type.Tensor,
    "Parameter": OrderedDict([
        ("name", Parameter_Type.Name),
        ("input_Tensor_Names", Parameter_Type.Tensor_list),
        ("tensor_Func", tf.add_n)
        ]),
    "Required": ["name", "input_Tensor_Names", "tensor_Func"],
    "Description": '''
Add tensors to input. All inserted tensors must have the same shape.
1) Name: An identifier within the process. This value does not affect learning.
2) input_Tensor_Name: Determines from which tensors the calculation is to proceed. This value is required.
'''
    }

func_Dict["Add"] = {
    "Func_Type": Func_Type.Fundamental_Math,
    "Tensor_Type": Tensor_Type.Tensor,
    "Parameter": OrderedDict([
        ("name", Parameter_Type.Name),
        ("input_Tensor_Names", Parameter_Type.Tensor2),
        ("tensor_Func", Compatible_Functions.Basic_Ops.add)
        ]),
    "Required": ["name", "input_Tensor_Names", "tensor_Func"],
    "Description": '''
Add two tensors to receive. Both tensors must have the same shape.
1) Name: An identifier within the process. This value does not affect learning.
2) input_Tensor_Name: Determines from which tensors the calculation is to proceed. This value is required.
'''
    }

func_Dict["Subtract"] = {
    "Func_Type": Func_Type.Fundamental_Math,
    "Tensor_Type": Tensor_Type.Tensor,
    "Parameter": OrderedDict([
        ("name", Parameter_Type.Name),
        ("input_Tensor_Names", Parameter_Type.Tensor2),
        ("tensor_Func", Compatible_Functions.Basic_Ops.subtract)
        ]),
    "Required": ["name", "input_Tensor_Names", "tensor_Func"],
    "Description": '''
Subtract the second tensor from the first tensor. Both tensors must have the same shape.
1) Name: An identifier within the process. This value does not affect learning.
2) input_Tensor_Name: Determines from which tensors the calculation is to proceed. This value is required.
'''
    }

func_Dict["Multiply"] = {
    "Func_Type": Func_Type.Fundamental_Math,
    "Tensor_Type": Tensor_Type.Tensor,
    "Parameter": OrderedDict([
        ("name", Parameter_Type.Name),
        ("input_Tensor_Names", Parameter_Type.Tensor2),
        ("tensor_Func", Compatible_Functions.Basic_Ops.multiply)
        ]),
    "Required": ["name", "input_Tensor_Names", "tensor_Func"],
    "Description": '''
The two input tensors are elementwise-multiply. Both tensors must have the same shape.
1) Name: An identifier within the process. This value does not affect learning.
2) input_Tensor_Name: Determines from which tensors the calculation is to proceed. This value is required.
'''
    }

func_Dict["Divide"] = {
    "Func_Type": Func_Type.Fundamental_Math,
    "Tensor_Type": Tensor_Type.Tensor,
    "Parameter": OrderedDict([
        ("name", Parameter_Type.Name),
        ("input_Tensor_Names", Parameter_Type.Tensor2),
        ("tensor_Func", Compatible_Functions.Basic_Ops.divide)
        ]),
    "Required": ["name", "input_Tensor_Names", "tensor_Func"],
    "Description": '''
Elementwise-divide the second tensor in the first tensor. Both tensors must have the same shape.
1) Name: An identifier within the process. This value does not affect learning.
2) input_Tensor_Name: Determines from which tensors the calculation is to proceed. This value is required.
'''
    }

func_Dict["Mean"] = {
    "Func_Type": Func_Type.Fundamental_Math,
    "Tensor_Type": Tensor_Type.Tensor,
    "Parameter": OrderedDict([
        ("name", Parameter_Type.Name),
        ("input_Tensor_Names", Parameter_Type.Tensor),
        ("axis", Parameter_Type.Positive_int_list),
        ("keepdims", Parameter_Type.Bool),
        ("tensor_Func", tf.reduce_mean)
        ]),
    "Required": ["name", "input_Tensor_Names", "tensor_Func", "axis"],
    "Description": '''
Calculate the mean value of the input tensor with respect to the specified axis.
1) Name: An identifier within the process. This value does not affect learning.
2) input_Tensor_Name: Determines from which tensor the calculation is to proceed. This value is required.
3) axis: Sets the axis on which to obtain the average.
4) keepdims: After determining the average, decide whether to keep the dimension of the tensor. If it is False, it becomes 'number of axis input - 1' dimension.
'''
    }

func_Dict["Sum"] = {
    "Func_Type": Func_Type.Fundamental_Math,
    "Tensor_Type": Tensor_Type.Tensor,
    "Parameter": OrderedDict([
        ("name", Parameter_Type.Name),
        ("input_Tensor_Names", Parameter_Type.Tensor),
        ("axis", Parameter_Type.Positive_int_list),
        ("keepdims", Parameter_Type.Bool),
        ("tensor_Func", tf.reduce_sum)
        ]),
    "Required": ["name", "input_Tensor_Names", "tensor_Func", "axis"],
    "Description": '''
Calculate the sum based on the specified axis of the input tensor. 
1) Name: An identifier within the process. This value does not affect learning.
2) input_Tensor_Name: Determines from which tensor the calculation is to proceed. This value is required.
3) axis: Sets the axis on which to obtain the sum.
4) keepdims: After determining the sum, decide whether to keep the dimension of the tensor. If it is False, it becomes 'number of axis input - 1' dimension.
'''
    }

func_Dict["Max"] = {
    "Func_Type": Func_Type.Fundamental_Math,
    "Tensor_Type": Tensor_Type.Tensor,
    "Parameter": OrderedDict([
        ("name", Parameter_Type.Name),
        ("input_Tensor_Names", Parameter_Type.Tensor),
        ("axis", Parameter_Type.Positive_int_list),
        ("keepdims", Parameter_Type.Bool),
        ("tensor_Func", tf.reduce_max)
        ]),
    "Required": ["name", "input_Tensor_Names", "tensor_Func", "axis"],
    "Description": '''
Calculate the max value based on the specified axis of the input tensor.
1) Name: An identifier within the process. This value does not affect learning.
2) input_Tensor_Name: Determines from which tensor the calculation is to proceed. This value is required.
3) axis: Sets the axis on which to obtain the max value.
4) keepdims: After determining the max value, decide whether to keep the dimension of the tensor. If it is False, it becomes 'number of axis input - 1' dimension.
'''
    }

func_Dict["Min"] = {
    "Func_Type": Func_Type.Fundamental_Math,
    "Tensor_Type": Tensor_Type.Tensor,
    "Parameter": OrderedDict([
        ("name", Parameter_Type.Name),
        ("input_Tensor_Names", Parameter_Type.Tensor),
        ("axis", Parameter_Type.Positive_int_list),
        ("keepdims", Parameter_Type.Bool),
        ("tensor_Func", tf.reduce_min)
        ]),
    "Required": ["name", "input_Tensor_Names", "tensor_Func", "axis"],
    "Description": '''
Calculate the min value based on the specified axis of the input tensor.
1) Name: An identifier within the process. This value does not affect learning.
2) input_Tensor_Name: Determines from which tensor the calculation is to proceed. This value is required.
3) axis: Sets the axis on which to obtain the min value.
4) keepdims: After determining the min value, decide whether to keep the dimension of the tensor. If it is False, it becomes 'number of axis input - 1' dimension.
'''
    }



func_Dict["Absolute_Difference_Loss"] = {
    "Func_Type": Func_Type.Loss_Calc,    #Abs loss
    "Tensor_Type": Tensor_Type.Loss,
    "Parameter": OrderedDict([
        ("name", Parameter_Type.Name),
        ("label_Tensor_Name", Parameter_Type.Tensor),
        ("prediction_Tensor_Name", Parameter_Type.Tensor),
        ("loss_Func", tf.losses.absolute_difference),
        ("weights", Parameter_Type.Positive_float)
        ]),
    "Required": ["name", "loss_Func", "label_Tensor_Name", "prediction_Tensor_Name"],
    "Description": '''
The absolute value of the difference between label and prediction (L1 loss). Label and prediction must have the same shape.
1) Name: An identifier within the process. This value does not affect learning.
2) label_Tensor_Name: Observation value that aims at learning. This value is required.
3) prediction_Tensor_Name: The predicted value calculated by the model. This value is required.
4) weights: Assign some weight to loss. 1 if not set.
'''
    }

func_Dict["Mean_Squared_Error_Loss"] = {
    "Func_Type": Func_Type.Loss_Calc,    #MSE loss
    "Tensor_Type": Tensor_Type.Loss,
    "Parameter": OrderedDict([
        ("name", Parameter_Type.Name),
        ("label_Tensor_Name", Parameter_Type.Tensor),
        ("prediction_Tensor_Name", Parameter_Type.Tensor),
        ("loss_Func", tf.losses.mean_squared_error),
        ("weights", Parameter_Type.Positive_float)
        ]),
    "Required": ["name", "loss_Func", "label_Tensor_Name", "prediction_Tensor_Name"],
    "Description": '''
The sum of the squares of the label and prediction squares (L2 loss). Label and prediction must have the same shape.
1) Name: An identifier within the process. This value does not affect learning.
2) label_Tensor_Name: Observation value that aims at learning. This value is required.
3) prediction_Tensor_Name: The predicted value calculated by the model. This value is required.
4) weights: Assign some weight to loss. 1 if not set.
'''
    }

func_Dict["Sigmoid_Loss"] = {
    "Func_Type": Func_Type.Loss_Calc,
    "Tensor_Type": Tensor_Type.Loss,
    "Parameter": OrderedDict([
        ("name", Parameter_Type.Name),
        ("label_Tensor_Name", Parameter_Type.Tensor),
        ("logit_Tensor_Name", Parameter_Type.Tensor),
        ("loss_Func", tf.losses.sigmoid_cross_entropy),
        ("weights", Parameter_Type.Positive_float)
        ]),
    "Required": ["name", "loss_Func", "label_Tensor_Name", "logit_Tensor_Name"],
    "Description": '''
Calculate sigmoid cross entropy loss of label and logit. Label and logit must have the same shape. ※Logit is the unscaled logit before applying sigmoid.
1) Name: An identifier within the process. This value does not affect learning.
2) label_Tensor_Name: Observation value that aims at learning. This value is required.
3) logit_Tensor_Name: The unscaled logit produced by the model. This value is required.
4) weights: Assign some weight to loss. 1 if not set.
'''
    }

func_Dict["Softmax_Loss"] = {
    "Func_Type": Func_Type.Loss_Calc,
    "Tensor_Type": Tensor_Type.Loss,
    "Parameter": OrderedDict([
        ("name", Parameter_Type.Name),
        ("label_Tensor_Name", Parameter_Type.Tensor),
        ("logit_Tensor_Name", Parameter_Type.Tensor),
        ("loss_Func", tf.losses.softmax_cross_entropy),
        ("weights", Parameter_Type.Positive_float)
        ]),
    "Required": ["name", "loss_Func", "label_Tensor_Name", "logit_Tensor_Name"],
    "Description": '''
Calculates Softmax cross entropy loss of label and logit. Label and logit must have the same shape. ※Logit is an unscaled logit before applying softmax.
1) Name: An identifier within the process. This value does not affect learning.
2) label_Tensor_Name: Observation value that aims at learning. This value is required.
3) logit_Tensor_Name: The unscaled logit produced by the model. This value is required.
4) weights: Assign some weight to loss. 1 if not set.
'''
    }

func_Dict["Sparse_Softmax_Loss"] = {
    "Func_Type": Func_Type.Loss_Calc,
    "Tensor_Type": Tensor_Type.Loss,
    "Parameter": OrderedDict([
        ("name", Parameter_Type.Name),
        ("label_Tensor_Name", Parameter_Type.Tensor),
        ("logit_Tensor_Name", Parameter_Type.Tensor),
        ("loss_Func", tf.losses.sparse_softmax_cross_entropy),
        ("weights", Parameter_Type.Positive_float)
        ]),
    "Required": ["name", "loss_Func", "label_Tensor_Name", "logit_Tensor_Name"],
    "Description": '''
Calculates softmax cross entropy loss of Label and logit. Label should be one dimension less than logit, and it must be an int32 dtype containing one of the indexes for the last dimension of logit (e.g. if logit shape = [B, 32, 50], label shape = [B, 32] and each value is from 0 to 49). ※Logit is an unscaled logit before applying softmax.
1) Name: An identifier within the process. This value does not affect learning.
2) label_Tensor_Name: Observation value that aims at learning. This value is required.
3) logit_Tensor_Name: The unscaled logit produced by the model. This value is required.
4) weights: Assign some weight to loss. 1 if not set.
'''
    }


func_Dict["ADAM_Optimizer"] = {
    "Func_Type": Func_Type.Optimizer,
    "Tensor_Type": Tensor_Type.Optimizer,
    "Parameter": OrderedDict([
        ("optimizer_Func", tf.train.AdamOptimizer),
        ("initial_learning_rate", Parameter_Type.Positive_float),
        ("beta1", Parameter_Type.Positive_float),
        ("beta2", Parameter_Type.Positive_float),
        ("epsilon", Parameter_Type.Positive_float),
        ("decay_method", Parameter_Type.Decay_method),
        ("decay_steps", Parameter_Type.Positive_int),
        ("decay_rate", Parameter_Type.Positive_float),        
        ("warmup_steps", Parameter_Type.Positive_int),
        ("applied_variable", Parameter_Type.Variable_list)
        ]),
    "Required": ["optimizer_Func"],
    "Description": '''
Create an optimizer using the ADAM algorithm.
1) initial_learning_rate: Determines the initial learning rate. If not set, 0.001 is assigned.
2) beta1: Set up beta1. If not set, 0.9 is assigned.
3) beta2: Set up beta2. If not set, 0.999 is assigned.
4) epsilon: Set up epsilon. If not set, 1e-8 is assigned.
5) decay_method: Sets the decay method of the learning rate. If not set, no decay.
6) decay_steps: One of the decay criteria when decay_method is exponential. When learning reaches each decay_step, learning rate decays by decay_rate than before. If not set, it is 0.5. If not exponential, this value is ignored.
7) decay_rate: One of the decay criteria when decay_method is exponential. When learning reaches each decay_step, learning rate decays by decay_rate than before. If not set, it is 0.5. If not exponential, this value is ignored.
8) warmup_steps: The decay criterion when decay_method is Noam. The learning rate gradually increases until the learning reaches warmup_steps and becomes initial_learning_rate at warmup_steps. After that, it gradually decays. If not Noam, this value is ignored.
9) applied_variable: A list of weights for which updates are in progress during training. The excluded weights are fixed in the process without updating.
'''
    }

func_Dict["GD_Optimizer"] = {
    "Func_Type": Func_Type.Optimizer,
    "Tensor_Type": Tensor_Type.Optimizer,
    "Parameter": OrderedDict([        
        ("optimizer_Func", tf.train.GradientDescentOptimizer),
        ("initial_learning_rate", Parameter_Type.Positive_float),
        ("decay_method", Parameter_Type.Decay_method),
        ("decay_steps", Parameter_Type.Positive_int),
        ("decay_rate", Parameter_Type.Positive_float),
        ("warmup_steps", Parameter_Type.Positive_int),
        ("applied_variable", Parameter_Type.Variable_list)
        ]),
    "Required": ["optimizer_Func"],
    "Description": '''
Create an optimizer using the gradient descent algorithm.
1) initial_learning_rate: Determines the initial learning rate. If not set, 0.001 is assigned.
2) decay_method: Sets the decay method of the learning rate. If not set, no decay.
3) decay_steps: One of the decay criteria when decay_method is exponential. When learning reaches each decay_step, learning rate decays by decay_rate than before. If not set, it is 0.5. If not exponential, this value is ignored.
4) decay_rate: One of the decay criteria when decay_method is exponential. When learning reaches each decay_step, learning rate decays by decay_rate than before. If not set, it is 0.5. If not exponential, this value is ignored.
5) warmup_steps: The decay criterion when decay_method is Noam. The learning rate gradually increases until the learning reaches warmup_steps and becomes initial_learning_rate at warmup_steps. After that, it gradually decays. If not Noam, this value is ignored.
6) applied_variable: A list of weights for which updates are in progress during training. The excluded weights are fixed in the process without updating.
'''
    }


func_Dict["Mean_Squared_Error"] = {
    "Func_Type": Func_Type.Test_Calc,
    "Tensor_Type": Tensor_Type.Tensor,
    "Parameter": OrderedDict([
        ("name", Parameter_Type.Name),
        ("input_Tensor_Names", Parameter_Type.Tensor2),
        ("tensor_Func", Compatible_Functions.Custom.mean_squared_error)
        ]),
    "Required": ["name", "input_Tensor_Names", "tensor_Func"],
    "Description": '''
Generates a mean squared error tensor of two input tensors.
1) Name: An identifier within the process. This value does not affect learning.
2) input_Tensor_Name: Determine from which tensors to proceed the calculation. This value is required.
'''
    }

func_Dict["Cross_Entropy"] = {
    "Func_Type": Func_Type.Test_Calc,
    "Tensor_Type": Tensor_Type.Tensor,
    "Parameter": OrderedDict([
        ("name", Parameter_Type.Name),
        ("input_Tensor_Names", Parameter_Type.Tensor2),
        ("tensor_Func", Compatible_Functions.Custom.cross_entropy)
        ]),
    "Required": ["name", "input_Tensor_Names", "tensor_Func"],
    "Description": '''
Creates a cross entropy tensor of the input two tensors. The first tensor is assigned to label and the second tensor is assigned to prediction.
1) Name: An identifier within the process. This value does not affect learning.
2) input_Tensor_Name: Determine from which tensors to proceed the calculation. This value is required.
'''
    }

func_Dict["Cosine_Similarity"] = {
    "Func_Type": Func_Type.Test_Calc,
    "Tensor_Type": Tensor_Type.Tensor,
    "Parameter": OrderedDict([
        ("name", Parameter_Type.Name),
        ("input_Tensor_Names", Parameter_Type.Tensor2),
        ("tensor_Func", Compatible_Functions.Custom.cosine_similarity)
        ]),
    "Required": ["name", "input_Tensor_Names", "tensor_Func"],
    "Description": '''
Creates a cosine similarity tensor of two input tensors.
1) Name: An identifier within the process. This value does not affect learning.
2) input_Tensor_Name: Determine from which tensors to proceed the calculation. This value is required.
'''
    }

func_Dict["Euclidean_Distance"] = {
    "Func_Type": Func_Type.Test_Calc,
    "Tensor_Type": Tensor_Type.Tensor,
    "Parameter": OrderedDict([
        ("name", Parameter_Type.Name),
        ("input_Tensor_Names", Parameter_Type.Tensor2),
        ("tensor_Func", Compatible_Functions.Custom.euclidean_distance)
        ]),
    "Required": ["name", "input_Tensor_Names", "tensor_Func"],
    "Description": '''
Creates a Euclidean distance tensor of two input tensors.
1) Name: An identifier within the process. This value does not affect learning.
2) input_Tensor_Name: Determine from which tensors to proceed the calculation. This value is required.
'''
    }

func_Dict["Multi_Test"] = {
    "Func_Type": Func_Type.Test_Calc,
    "Tensor_Type": Tensor_Type.Tensor,
    "Parameter": OrderedDict([
        ("name", Parameter_Type.Name),
        ("input_Tensor_Names", Parameter_Type.Tensor2),
        ("tensor_Func", Compatible_Functions.Custom.multi_test_calc)
        ]),
    "Required": ["name", "input_Tensor_Names", "tensor_Func"],
    "Description": '''
It generates the mean squared error, cross entropy, cosine similarity, and Euclidean distance tensor of each input tensor. For a cross entropy calculation, the first tensor is assigned to label and the second tensor is assigned to prediction.
1) Name: An identifier within the process. This value does not affect learning.
2) input_Tensor_Name: Determine from which tensors to proceed the calculation. This value is required.
'''
    }

func_Dict["Arg_Max"] = {
    "Func_Type": Func_Type.Test_Calc,
    "Tensor_Type": Tensor_Type.Tensor,
    "Parameter": OrderedDict([
        ("name", Parameter_Type.Name),
        ("input_Tensor_Names", Parameter_Type.Tensor),
        ("tensor_Func", Compatible_Functions.Custom.argmax)
        ]),
    "Required": ["name", "input_Tensor_Names", "tensor_Func"],
    "Description": '''
Creates a tensor that displays the index with the highest value based on the last dimension of the input tensor.
1) Name: An identifier within the process. This value does not affect learning.
2) input_Tensor_Name: Determines from which tensor the calculation is to proceed. This value is required.
'''
    }

func_Dict["Semantic_Stress"] = {
    "Func_Type": Func_Type.Test_Calc,
    "Tensor_Type": Tensor_Type.Tensor,
    "Parameter": OrderedDict([
        ("name", Parameter_Type.Name),
        ("input_Tensor_Names", Parameter_Type.Tensor),
        ("tensor_Func", Compatible_Functions.Custom.semantic_stress)
        ]),
    "Required": ["name", "input_Tensor_Names", "tensor_Func"],
    "Description": '''
Generates the semantic stress of input tensor. Please refer to the following article on the specific definition of semantic stress:
Plaut, D. C. (1997). Structure and function in the lexical system: Insights from distributed models of word reading and lexical decision. Language and cognitive processes, 12(5-6), 765-806.
1) Name: An identifier within the process. This value does not affect learning.
2) input_Tensor_Name: Determines from which tensor the calculation is to proceed. This value is required.
'''
    }

func_Dict["Ones"] = {
    "Func_Type": Func_Type.Create,
    "Tensor_Type": Tensor_Type.Create,
    "Parameter": OrderedDict([
        ("name", Parameter_Type.Name),
        ("shape", Parameter_Type.Positive_int_list),
        ("tensor_Func", tf.ones)
        ]),
    "Required": ["name", "shape", "tensor_Func"],
    "Description": '''
Generates a tensor with all values equal to 1.
1) Name: An identifier within the process. This value does not affect learning.
2) shape: Sets the shape of the tensor to be created. This value is required.
'''
    }

func_Dict["Zeros"] = {
    "Func_Type": Func_Type.Create,
    "Tensor_Type": Tensor_Type.Create,
    "Parameter": OrderedDict([
        ("name", Parameter_Type.Name),
        ("shape", Parameter_Type.Positive_int_list),
        ("tensor_Func", tf.zeros)
        ]),
    "Required": ["name", "shape", "tensor_Func"],
    "Description": '''
Generates a tensor with all values equal to 0.
1) Name: An identifier within the process. This value does not affect learning.
2) shape: Sets the shape of the tensor to be created. This value is required.
'''
    }

func_Dict["Random_Normal"] = {
    "Func_Type": Func_Type.Create,
    "Tensor_Type": Tensor_Type.Create,
    "Parameter": OrderedDict([
        ("name", Parameter_Type.Name),
        ("shape", Parameter_Type.Positive_int_list),
        ("mean", Parameter_Type.Float),
        ("stddev", Parameter_Type.Positive_float),
        ("tensor_Func", tf.random_normal)
        ]),
    "Required": ["name", "shape", "mean", "stddev", "tensor_Func"],
    "Description": '''
Generate tensor according to normal distribution.
1) Name: An identifier within the process. This value does not affect learning.
2) shape: Sets the shape of the tensor to be created. This value is required.
3) mean: Sets the mean of the normal distribution. If not set, 0 is assigned.
4) stddev: Sets the standard deviation of the normal distribution. If not set, 1 is assigned.
'''
    }

func_Dict["Random_Uniform"] = {
    "Func_Type": Func_Type.Create,
    "Tensor_Type": Tensor_Type.Create,
    "Parameter": OrderedDict([
        ("name", Parameter_Type.Name),
        ("shape", Parameter_Type.Positive_int_list),
        ("minval", Parameter_Type.Float),
        ("maxval", Parameter_Type.Float),
        ("tensor_Func", tf.random_uniform)
        ]),
    "Required": ["name", "shape", "minval", "maxval", "tensor_Func"],
    "Description": '''
Generate tensor according to uniform distribution.
1) Name: An identifier within the process. This value does not affect learning.
2) shape: Sets the shape of the tensor to be created. This value is required.
3) minval: Sets the minimum value of the tensor to be created. This value is required.
4) maxval: Sets the maximum value of the tensor to be generated. This value is required.
'''
    }


func_Dict["Clip"] = {
    "Func_Type": Func_Type.Other,
    "Tensor_Type": Tensor_Type.Tensor,
    "Parameter": OrderedDict([
        ("name", Parameter_Type.Name),
        ("input_Tensor_Names", Parameter_Type.Tensor),
        ("clip_value_min", Parameter_Type.Float),
        ("clip_value_max", Parameter_Type.Float),
        ("tensor_Func", tf.clip_by_value)
        ]),
    "Required": ["name", "input_Tensor_Names", "tensor_Func", "clip_value_min", "clip_value_max"],
    "Description": '''
It clips the value outside the given range of input tensor. Values lower than min are fixed at min and values higher than max are fixed at max.
1) Name: An identifier within the process. This value does not affect learning.
2) input_Tensor_Name: Determines from which tensor the calculation is to proceed. This value is required.
3) clip_value_min: Determines the min value. This value is required.
4) clip_value_max: Determines the max value. This value is required.
'''
    }
  
func_Dict["Dropout"] = {
    "Func_Type": Func_Type.Other,
    "Tensor_Type": Tensor_Type.Tensor,
    "Parameter": OrderedDict([
        ("name", Parameter_Type.Name),
        ("input_Tensor_Names", Parameter_Type.Tensor),
        ("rate", Parameter_Type.Positive_float),
        ("tensor_Func", tf.layers.dropout)
        ]),
    "Required": ["name", "input_Tensor_Names", "tensor_Func", "rate"],
    "Description": '''
Make each cell of input tensor zero at the specified probability rate. At the same time, the tensor is multiplied by 1 / rate. In Test, this function is not applied (output is same to input tensor)
1) Name: An identifier within the process. This value does not affect learning.
2) input_Tensor_Name: Determines from which tensor the calculation is to proceed. This value is required.
3) rate: Sets the probability that dropout will be applied. If not set, 0.5 is assigned.
'''
    }

func_Dict["Indexing"] = {
    "Func_Type": Func_Type.Other,
    "Tensor_Type": Tensor_Type.Tensor,
    "Parameter": OrderedDict([
        ("name", Parameter_Type.Name),
        ("input_Tensor_Names", Parameter_Type.Tensor),
        ("axis", Parameter_Type.Positive_int),
        ("index", Parameter_Type.Non_negative_int),
        ("tensor_Func", Compatible_Functions.Custom.indexing)
        ]),
    "Required": ["name", "input_Tensor_Names", "tensor_Func", "axis", "index"],
    "Description": '''
Extracts only the specified index of the specified axis. In case of axis 2 and index 3 in the tensor of Shape [B,3,5,7], a tensor of shape [B,3,7] of [:,:, 3,:] of the input tensor is generated.
1) Name: An identifier within the process. This value does not affect learning.
2) input_Tensor_Name: Determines from which tensor the calculation is to proceed. This value is required.
3) axis: Decide on which axis to extract. This value is required.
3) index: Decide which index of the set axis to extract. This value is required.
'''
    }

func_Dict["Embedding"] = {
    "Func_Type": Func_Type.Other,
    "Tensor_Type": Tensor_Type.Tensor,
    "Parameter": OrderedDict([
        ("name", Parameter_Type.Name),
        ("input_Tensor_Names", Parameter_Type.Tensor),
        ("id_size", Parameter_Type.Positive_int),
        ("embedding_size", Parameter_Type.Positive_int),
        ("tensor_Func", Compatible_Functions.Custom.embedding),
        ("reuse_Tensor_Name", Parameter_Type.Process_and_Tensor_with_None)
        ]),
    "Required": ["name", "input_Tensor_Names", "tensor_Func", "id_size", "embedding_size"],
    "Description": '''
Generates an n+1 dimensional float tensor for an n dimensional int tensor that receives input. Within a single model, the same int value will always be the same float vector.
1) Name: An identifier within the process. This value does not affect learning.
2) input_Tensor_Name: Determines from which tensor the calculation is to proceed. Must be an int tensor. This value is required.
3) id_size: Sets the value range of the int tensor to be input. This value is required.
4) embedding_size: Sets the size of the float vector to be created. This value is required.
5) reuse_Tensor_Name: Import embedding variable from another embedding. In this case, both embeddings produce the same float vector for the same int value.
'''
    }

func_Dict["Noise_Normal"] = {
    "Func_Type": Func_Type.Other,
    "Tensor_Type": Tensor_Type.Tensor,
    "Parameter": OrderedDict([
        ("name", Parameter_Type.Name),
        ("input_Tensor_Names", Parameter_Type.Tensor),
        ("mean", Parameter_Type.Float),
        ("stddev", Parameter_Type.Positive_float),
        ("tensor_Func", Compatible_Functions.Custom.add_noise_normal)
        ]),
    "Required": ["name", "input_Tensor_Names", "tensor_Func"],
    "Description": '''
Add Gaussian noise to the input tensor.
1) Name: An identifier within the process. This value does not affect learning.
2) input_Tensor_Name: Determines from which tensor the calculation is to proceed. This value is required.
3) mean: Sets the mean of the Gaussian noise. If not set, 0 is assigned.
4) stddev: Sets the standard deviation of the Gaussian noise. If not set, 1 is assigned.
'''
    }

func_Dict["Noise_Uniform"] = {
    "Func_Type": Func_Type.Other,
    "Tensor_Type": Tensor_Type.Tensor,
    "Parameter": OrderedDict([
        ("name", Parameter_Type.Name),
        ("input_Tensor_Names", Parameter_Type.Tensor),
        ("minval", Parameter_Type.Float),
        ("maxval", Parameter_Type.Float),
        ("tensor_Func", Compatible_Functions.Custom.add_noise_uniform)
        ]),
    "Required": ["name", "input_Tensor_Names", "minval", "maxval", "tensor_Func"],
    "Description": '''
Add uniform noise to the input tensor.
1) Name: An identifier within the process. This value does not affect learning.
2) input_Tensor_Name: Determines from which tensor the calculation is to proceed. This value is required.
3) minval: Sets the minimum value of noise. This value is required.
4) maxval: Sets the maximum value of noise. This value is required.
'''
    }

func_Dict["Batch_Normalization"] = {
    "Func_Type": Func_Type.Other,
    "Tensor_Type": Tensor_Type.Tensor,
    "Parameter": OrderedDict([
        ("name", Parameter_Type.Name),
        ("input_Tensor_Names", Parameter_Type.Tensor),        
        ("momentum", Parameter_Type.Positive_float),
        ("epsilon", Parameter_Type.Positive_float),
        ("tensor_Func", tf.layers.batch_normalization),        
        ("reuse_Tensor_Name", Parameter_Type.Process_and_Tensor_with_None)
        ]),
    "Required": ["name", "input_Tensor_Names", "tensor_Func"],
    "Description": '''
Generate a tensor that performs batch normalization on the last dimension of the input tensor. For more details on batch normalization see the document on 'tf.layers.batch_normalization' by tensorflow and the following paper:
Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift. arXiv preprint arXiv:1502.03167.
1) Name: An identifier within the process. This value does not affect learning.
2) input_Tensor_Name: Determines from which tensor the calculation is to proceed. This value is required.
3) momentum: Sets the momentum. If not set, 0.99 is assigned.
4) epsilon: Set epsilon. If not set, 0.001 is assigned.
5) reuse_Tensor_Name: Import weight from another batch normalization. In this case, the weight is shared with another layer.
'''
    }

func_Dict["Max_Pooling1d"] = {
    "Func_Type": Func_Type.Other,
    "Tensor_Type": Tensor_Type.Tensor,
    "Parameter": OrderedDict([
        ("name", Parameter_Type.Name),
        ("input_Tensor_Names", Parameter_Type.Tensor),
        ("tensor_Func", tf.layers.max_pooling1d),
        ("pool_size", Parameter_Type.Positive_int),
        ("strides", Parameter_Type.Positive_int),
        ("padding", Parameter_Type.Padding)
        ]),
    "Required": ["name", "input_Tensor_Names", "tensor_Func", "pool_size", "strides"],
    "Description": '''
Max pooling is performed on the input 3d tensor.
1) Name: An identifier within the process. This value does not affect learning.
2) input_Tensor_Names: Determines from which tensor the calculation is to proceed. The shape of the input tensor must be 3D in the form of [B, step, channel]. This value is required.
3) pool_size: Determines the size of the pooling window. This value is required.
5) strides: Determines the size of the movement of the pooling window. This value is required.
6) padding: After pooling through padding, determine the shape of the tensor is equal to the entered tensor.
'''
    }

func_Dict["Max_Pooling2d"] = {
    "Func_Type": Func_Type.Other,
    "Tensor_Type": Tensor_Type.Tensor,
    "Parameter": OrderedDict([
        ("name", Parameter_Type.Name),
        ("input_Tensor_Names", Parameter_Type.Tensor),
        ("tensor_Func", tf.layers.max_pooling2d),
        ("pool_size", Parameter_Type.Positive_int_list),
        ("strides", Parameter_Type.Positive_int_list),
        ("padding", Parameter_Type.Padding)
        ]),
    "Required": ["name", "input_Tensor_Names", "tensor_Func", "pool_size", "strides"],
    "Description": '''
Max pooling is performed on the input 4d tensor.
1) Name: An identifier within the process. This value does not affect learning.
2) input_Tensor_Names: Determines from which tensor the calculation is to proceed. The shape of the input tensor must be 4D in the form of [B, height, width, channel]. This value is required.
3) pool_size: Determines the size of the pooling window. This value is required.
5) strides: Determines the size of the movement of the pooling window. This value is required.
6) padding: After pooling through padding, determine the shape of the tensor is equal to the entered tensor.
'''
    }