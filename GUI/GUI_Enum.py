from enum import Enum;

class Func_Type(Enum):
    Placeholder = 0,
    Forward = 1,
    RNN = 2,
    Activation_Calc = 3,
    Reshape = 4,
    Fundamental_Math = 5,
    Loss_Calc = 6,
    Optimizer = 7,
    Test_Calc = 8,
    Create = 9,
    Other = 10,

class Parameter_Type(Enum):
    Name = 0,   #lineedit
    Dtype = 1,  #Preset combobox
    Int = 2,    #lineedit
    Positive_int = 3,    #lineedit
    Non_negative_int = 4,    #lineedit
    Int_list = 5,   #lineedit
    Positive_int_list = 6,   #lineedit
    Non_negative_int_list = 7,   #lineedit
    Float = 8,   #lineedit
    Positive_float = 9,   #lineedit
    Float_list = 10,   #lineedit    
    Positive_float_list = 11,   #lineedit
    Bool = 12,   #Preset combobox    
    Padding = 13,   #Preset combobox
    Decay_method = 14,  #Preset combobox
    Tensor = 15, #Combobox
    Tensor_with_None = 16,   #combobox
    Tensor2 = 17,    #2 Combobox
    Tensor_list = 18,   #Listwidget, combobox
    Process_and_Tensor = 19, #Combobox
    Process_and_Tensor_with_None = 20, #Combobox
    Process_and_Tensor2 = 21,    #2 Combobox
    Process_and_Tensor_list = 22,   #Listwidget, combobox
    Variable_list = 23,   #Listwidget, combobox
    Process_and_Variable_list = 24,   #Listwidget, combobox
    
    Hidden_activation_func = 25,
    Output_activation_func = 26,
    RNN_cell_func = 27,
    
    
    