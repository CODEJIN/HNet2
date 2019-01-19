import _pickle as pickle;
import tensorflow as tf;
import numpy as np;
import os;
from HNet2_Core import HNet;
from HNet_Enum import Tensor_Type, Learning_Rate_Decay_Method;

front_Caution = '''
### Cautions to use script
###
### 1) Set the pattern path. The '#Pattern' comment following code is the part that assign pattern.
### Since the pattern file is not assigned in the parameter, the user must manually add the pattern path to the 
### 'pattern_Path =' parameter. Enter the path to the  pattern you used to create the setting for the first
### time.
### 2) If you modify the process, you may not be able to use the checkpoint. Modifications to the process may
### change the shape of some or all of the weights used by the model, in which case checkpoints are not
### compatible.
### 3) Once loaded and the train screen is presented, check the process and pattern tabs before you start
### learning. If the structure and the pattern you modify are logically incorrect, in most cases an error
### will occur during loading and will not be passed. However, to prevent for an error, you may need to
### verify it yourself.
'''

def Generate_Script_Full(process_Name_List, graph_History, pattern_Dict, learning_List, test_Dict, export_Path, pattern_Export=True):
    process_Script_List = Generate_Script_Process(process_Name_List, graph_History);
    pattern_Script_List = Generate_Script_Pattern(pattern_Dict, export_Path if pattern_Export else None);
    learning_Script_List = Generate_Script_Learning(learning_List, test_Dict);

    export_List = [];    
    export_List.append(front_Caution);
    export_List.append('');
    export_List.append('');
    export_List.append('');
    export_List.append('#Process');
    export_List.extend(process_Script_List);
    export_List.append('');
    export_List.append('hNet.Run_Pattern_Manager()');
    export_List.append('');
    export_List.append('#Pattern');
    export_List.extend(pattern_Script_List);
    export_List.append('');
    export_List.append('hNet.Run_Learning_Manager()');
    export_List.append('');
    export_List.append('#Learning');
    export_List.extend(learning_Script_List);
    export_List.append('');
    export_List.append('hNet.learning_Manager.Lock()');    

    if not export_Path is None:
        with open(export_Path, 'w') as f:
            f.write('\n'.join(export_List));

def Generate_Script_Process(process_Name_List, graph_History): 
    export_List = [];
    for process_Name in process_Name_List:
        export_List.append('hNet.process_Manager.Set_New_Process(\'{}\')'.format(process_Name));
    export_List.append('');
    for tensor_Type, parameter_Dict in graph_History:        
        if tensor_Type == Tensor_Type.Placeholder:
            export_List.append('hNet.process_Manager.Placeholder_Generate(process_Name= \'{0}\', name= \'{1}\', dtype= {2}, shape= {3})'.format(
                parameter_Dict['process_Name'],
                parameter_Dict['name'],
                'tf.{}'.format(parameter_Dict['dtype'].name),
                parameter_Dict['shape']
                ))

        elif tensor_Type == Tensor_Type.Tensor:
            parameter_List = [];
            for key, value in parameter_Dict.items():
                if key == 'tensor_Func':
                    value = 'tf.{}'.format(value._tf_api_names[0]) if hasattr(value, '_tf_api_names') else '{}.{}'.format(value.__module__, value.__name__);
                    #Some functions like 'softmax' do not decorated yet in tensorflow code. The following line is temporal bug fix.
                    value = value.replace('tensorflow.python.ops.nn_ops.', 'tf.nn.');
                elif type(value) == tf.DType:
                    value = 'tf.{}'.format(value.name);
                elif type(value) == str:
                    value = '\'{}\''.format(value);
                parameter_List.append('{}= {}'.format(key, value));
            export_List.append('hNet.process_Manager.Tensor_Generate({0})'.format(', '.join(parameter_List)));

        elif tensor_Type == Tensor_Type.RNN_Tensor:            
            parameter_List = [];
            for key, value in parameter_Dict.items():
                if key == 'rnn_Cell_Func':
                    value = 'tf.{}'.format(value._tf_api_names[0]) if hasattr(value, '_tf_api_names') else '{}.{}'.format(value.__module__, value.__name__);
                elif key == 'state_Func':
                    value = ('tf.{}'.format(value._tf_api_names[0]) if hasattr(value, '_tf_api_names') else '{}.{}'.format(value.__module__, value.__name__)) if not value is None else None;
                elif type(value) == tf.DType:
                    value = 'tf.{}'.format(value.name);
                elif type(value) == str:
                    value = '\'{}\''.format(value);
                parameter_List.append('{}= {}'.format(key, value));
            export_List.append('hNet.process_Manager.RNN_Tensor_Generate({0})'.format(', '.join(parameter_List)));

        elif tensor_Type == Tensor_Type.Create:
            parameter_List = [];
            for key, value in parameter_Dict.items():
                if key == 'tensor_Func':
                    value = 'tf.{}'.format(value._tf_api_names[0]) if hasattr(value, '_tf_api_names') else '{}.{}'.format(value.__module__, value.__name__);
                elif type(value) == tf.DType:
                    value = 'tf.{}'.format(value.name);
                elif type(value) == str:
                    value = '\'{}\''.format(value);
                parameter_List.append('{}= {}'.format(key, value));
            export_List.append('hNet.process_Manager.Create_Tensor_Generate({0})'.format(', '.join(parameter_List)));

        elif tensor_Type == Tensor_Type.Loss:
            export_List.append('hNet.process_Manager.Loss_Generate(process_Name= \'{0}\', name= \'{1}\', label_Tensor_Name= \'{2}\', loss_Func= {3}, logit_Tensor_Name= {4}, prediction_Tensor_Name= {5}, weights= {6})'.format(
                parameter_Dict['process_Name'],
                parameter_Dict['name'],
                parameter_Dict['label_Tensor_Name'],
                'tf.{}'.format(parameter_Dict['loss_Func']._tf_api_names[0]) if hasattr(parameter_Dict['loss_Func'], '_tf_api_names') else '{}.{}'.format(parameter_Dict['loss_Func'].__module__, parameter_Dict['loss_Func'].__name__),
                '\'{}\''.format(parameter_Dict['logit_Tensor_Name']) if not parameter_Dict['logit_Tensor_Name'] is None else parameter_Dict['logit_Tensor_Name'],
                '\'{}\''.format(parameter_Dict['prediction_Tensor_Name']) if not parameter_Dict['prediction_Tensor_Name'] is None else parameter_Dict['prediction_Tensor_Name'],
                parameter_Dict['weights']
                ))

        elif tensor_Type == Tensor_Type.Optimizer:
            parameter_List = [];
            for key, value in parameter_Dict.items():                
                if key == 'optimizer_Func':
                    value = 'tf.{}'.format(value._tf_api_names[0]) if hasattr(value, '_tf_api_names') else '{}.{}'.format(value.__module__, value.__name__)
                elif type(value) == tf.DType:
                    value = 'tf.{}'.format(value.name);
                elif type(value) == str:
                    value = '\'{}\''.format(value);
                parameter_List.append('{}= {}'.format(key, value));
            export_List.append('hNet.process_Manager.Optimizer_Generate({0})'.format(', '.join(parameter_List)));

    export_List.append('');
    export_List.append('hNet.process_Manager.Tensor_Initialize()');

    return export_List;

def Generate_Script_Pattern(pattern_Dict, pattern_Export_Path= None):
    export_List = [];

    if not pattern_Export_Path is None:
        pattern_Export_Path_Dir = os.path.join(os.path.dirname(pattern_Export_Path), 'Pattern_Pickle').replace('\\', '/');
        os.makedirs(pattern_Export_Path_Dir, exist_ok=True);

    for pattern_Name, pattern in pattern_Dict.items():
        if not pattern_Export_Path is None:
            pattern_Path = os.path.join(pattern_Export_Path_Dir, '{}.pickle'.format(pattern_Name)).replace('\\', '/');
            with open(pattern_Path, 'wb') as f:
                pickle.dump(pattern, f, protocol=2);
        else:
            pattern_Path = '';

        export_List.append('hNet.pattern_Manager.Pattern_Load(pattern_Name= \'{0}\', pattern_Path= \'{1}\')'.format(pattern_Name, pattern_Path));

    return export_List;
        
def Generate_Script_Learning(learning_List, test_Dict):
    export_List = [];

    for learning_Info in learning_List:
        export_List.append('hNet.learning_Manager.Set_New_Learning(');
        export_List.append('    learning_Name= \'{}\', '.format(learning_Info['Name']));
        export_List.append('    epoch= {}, '.format(learning_Info['Epoch']));
        export_List.append('    checkpoint_Save_Interval= {}, '.format(learning_Info['Checkpoint_Save_Interval']));
        export_List.append('    test_Interval= {}, '.format(learning_Info['Test_Interval']));
        export_List.append('    mini_Batch= {}, '.format(learning_Info['Mini_Batch']));
        export_List.append('    use_Probability= {}'.format(learning_Info['Use_Probability']));
        export_List.append('    )');
        export_List.append('');

        for matching_Info in learning_Info['Matching_Info_List']:
            export_List.append('hNet.learning_Manager.Learning_Add_Mathing_Info(');
            export_List.append('    learning_Name= \'{}\', '.format(learning_Info['Name']));
            export_List.append('    matching_Info= hNet.learning_Manager.Get_Matching_Info(');
            export_List.append('        process_Name= \'{}\', '.format(matching_Info['Process']));
            export_List.append('        pattern_Name= \'{}\', '.format(matching_Info['Pattern']));
            export_List.append('        matching_List= [');
            for placeholder_Name, pattern_Column_Name in matching_Info['Matching_List']:
                export_List.append('            (\'{}\', \'{}\'),'.format(placeholder_Name, pattern_Column_Name));
            export_List.append('            ],');
            export_List.append('        probability_Pattern_Column_Name= {}'.format('\'{}\''.format(matching_Info['Probability']) if not matching_Info['Probability'] is None else None));
            export_List.append('        )[1]');
            export_List.append('    )');
            export_List.append('');

    for test_Name, test_Info in test_Dict.items():
        export_List.append('hNet.learning_Manager.Set_New_Test(');
        export_List.append('    test_Name= \'{}\','.format(test_Name));
        export_List.append('    mini_Batch= {},'.format(test_Info['Mini_Batch']));
        export_List.append('    matching_Info = hNet.learning_Manager.Get_Matching_Info(');
        export_List.append('        process_Name= \'{}\','.format(test_Info['Matching_Info']['Process']));
        export_List.append('        pattern_Name= \'{}\','.format(test_Info['Matching_Info']['Pattern']));
        export_List.append('        matching_List= [');
        for placeholder_Name, pattern_Column_Name in test_Info['Matching_Info']['Matching_List']:
                export_List.append('            (\'{}\', \'{}\'),'.format(placeholder_Name, pattern_Column_Name));        
        export_List.append('            ],');
        export_List.append('        get_Tensor_Name_List = [{}]'.format(', '.join(['\'{}\''.format(tensor_Name) for tensor_Name in test_Info['Get_Tensor_Name_List']])));
        export_List.append('        )[1],');
        export_List.append('    get_Tensor_Name_List = [{}],'.format(', '.join(['\'{}\''.format(tensor_Name) for tensor_Name in test_Info['Get_Tensor_Name_List']])));
        export_List.append('    attached_Pattern_Column_Name_List = [{}]'.format(', '.join(['\'{}\''.format(tensor_Name) for tensor_Name in test_Info['Attached_Pattern_Column_Name_List']])));
        export_List.append('    )');
        export_List.append('');

    return export_List;

def Pickle_to_Script(pickle_File_Path):
    with open(pickle_File_Path, 'rb') as f:
        load_Dict = pickle.load(f);
    
    export_Path = os.path.join(os.path.dirname(pickle_File_Path), 'Prerequisite.txt').replace('\\', '/');

    Generate_Script_Full(
        load_Dict['Process_Name_List'],
        load_Dict['Graph_History'],
        load_Dict['Pattern_Dict'],
        load_Dict['Learning_List'],
        load_Dict['Test_Dict'],
        export_Path,
        True
        )

if __name__ == '__main__':
    Pickle_to_Script('F:/Test/Model/Prerequisite.pickle')