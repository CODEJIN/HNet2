import numpy as np;
import tensorflow as tf;
import os, sys, inspect, collections, types;
import _pickle as pickle;
import networkx as nx

from HNet_Enum import Tensor_Type, Learning_Rate_Decay_Method;

class Process_Manager:
    def Check_Lock(func):
        def wrapper(self, *args, **kwargs):
            if self.is_Locked:
                return False, Exception("Process modifying is locked.")
            return func(self, *args, **kwargs)
        return wrapper

    def Check_Optimizer(func):
        def wrapper(self, *args, **kwargs):
            if "{}.Optimizer".format(kwargs['process_Name']) in self.tensor_Dict[kwargs['process_Name']].keys():
                return False, Exception("An optimizer is already assigned to process '{}'. This function cannot use in this process anymore.".format(kwargs['process_Name']));
            return func(self, *args, **kwargs)
        return wrapper

    def __init__(self, tf_Session):
        self.tf_Session = tf_Session;

        self.graph_History = [];
        
        self.scope_Dict = {};

        self.tensor_Dict = {};  #[process_Name][tensor_Name]
        self.placeholder_Dict = {};
        self.rnn_Cell_Dict = {};    #[process_Name][tensor_Name]
        self.rnn_State_Assign_List = [];
        self.create_Dict = {}
        self.loss_Dict = {};    #[process_Name][tensor_Name]

        self.reuse_Reference_Dict = {};  #[process_Name][tensor_Name]
        self.tensor_Dependency_Dict = {};   #[process_Name][tensor_Name]
        self.tensor_Connection_Dict = {};   #[process_Name]

        self.global_Tensor_Dict ={
            "Variable_Dict": {},
            "Placeholder_Dict": {},
            "Assign_Tensor_Dict": {},
            }
        for variable_Name in ['global_Step', 'global_Epoch', 'learning_Flow_Index', 'local_Step', 'local_Epoch']:
            self.global_Tensor_Dict['Variable_Dict'][variable_Name] = tf.Variable(0, name='global_Step', trainable = False);
            self.global_Tensor_Dict['Placeholder_Dict'][variable_Name] = tf.placeholder(tf.int32);
            self.global_Tensor_Dict['Assign_Tensor_Dict'][variable_Name] = tf.assign(
                self.global_Tensor_Dict["Variable_Dict"][variable_Name],
                self.global_Tensor_Dict["Placeholder_Dict"][variable_Name]
                )
        self.is_Training_Placeholder = tf.placeholder(tf.bool, name="is_training");

        self.is_Locked = False;
            
    def Get_Tensor_Name_List(self, process_Name = None):
        if process_Name is None:
            return list(set([(process_Name, name) for process_Name, process in self.tensor_Dict.items() for name in process.keys()]));
        else:
            return [name for name in self.tensor_Dict[process_Name].keys()];

    def Get_RNN_Cell_Name_List(self, process_Name = None):
        if process_Name is None:
            return list(set([name for process in self.rnn_Cell_Dict.values() for name in process.keys()]));
        else:
            return [name for name in self.rnn_Cell_Dict[process_Name].keys()];

    @Check_Lock
    def Set_New_Process(self, process_Name):
        try:
            if process_Name in self.tensor_Dict.keys():
                raise ValueError("There is already process with named '{0}'.".format(process_Name))

            with tf.variable_scope(process_Name) as new_Scope:
                self.scope_Dict[process_Name] = new_Scope

            self.tensor_Dict[process_Name] = {};
            self.placeholder_Dict[process_Name] = {};
            self.rnn_Cell_Dict[process_Name] = {};
            self.create_Dict[process_Name] = {};
            self.loss_Dict[process_Name] = {};

            self.reuse_Reference_Dict[process_Name] = {};

            self.tensor_Dependency_Dict[process_Name] = {};

            self.tensor_Connection_Dict[process_Name] = [];

            return True, None;

        except ValueError as e:            
            return False, e;

    @Check_Lock
    def Placeholder_Generate(self, process_Name, name, dtype, shape):        
        try:
            if name in self.Get_Tensor_Name_List():
                raise ValueError("There is already a tensor with named '{}'.".format(name))
            
            with tf.variable_scope(self.scope_Dict[process_Name]) as scope:
                with tf.name_scope(scope.original_name_scope):
                    new_Placeholder = tf.placeholder(dtype=dtype, shape=[None] + [x for x in shape], name=name);

            self.tensor_Dict[process_Name][name] = new_Placeholder;
            self.placeholder_Dict[process_Name][name] = new_Placeholder;

            self.graph_History.append((Tensor_Type.Placeholder, {"process_Name": process_Name, "name": name, "dtype": dtype, "shape": shape}));
            self.tensor_Dependency_Dict[process_Name][name] = [name];
            self.tensor_Connection_Dict[process_Name].append((name, name))

            return True, self.Get_Tensor_Shape(process_Name, name);

        except ValueError as e:            
            return False, e;
        except TypeError as e:            
            return False, e;

    @Check_Lock
    def Tensor_Generate(self, process_Name, name, input_Tensor_Names, tensor_Func, reuse_Tensor_Name=None, **kwargs):
        '''
        input_Tensor_Names: str list or str
        Don't use 'reuse' parameter directly. Use 'reuse_Tensor_Name'.
        '''        
        try:            
            if type(input_Tensor_Names) == str:
                input_Tensor_Name_List = [input_Tensor_Names]
            else:
                input_Tensor_Name_List = input_Tensor_Names
                            
            for input_Tensor_Name in input_Tensor_Name_List:
                if not input_Tensor_Name in self.Get_Tensor_Name_List(process_Name):
                    raise ValueError("There is no tensor with named '{0}' in process '{1}'.".format(input_Tensor_Name, process_Name));
            if name in self.Get_Tensor_Name_List(process_Name):
                raise ValueError("There is already tensor with named '{0}' in process '{1}'.".format(name, process_Name))            
            elif not reuse_Tensor_Name is None and not reuse_Tensor_Name[1] in self.reuse_Reference_Dict[reuse_Tensor_Name[0]].keys():
                raise ValueError("Tensor '{0}' in process '{1}' does not support 'reuse'.".format(*reuse_Tensor_Name))
            
            if len(input_Tensor_Name_List) == 1:
                input_Tensor = self.tensor_Dict[process_Name][input_Tensor_Name_List[0]];
            else:
                input_Tensor = [self.tensor_Dict[process_Name][input_Tensor_Name] for input_Tensor_Name in input_Tensor_Name_List]
                            
            input_Dict = {};
            if "training" in inspect.getargspec(tensor_Func).args and not "training" in kwargs: #Dropout or batch normalization
                input_Dict.update({"training": self.is_Training_Placeholder});

            if not reuse_Tensor_Name is None and reuse_Tensor_Name in self.Get_Tensor_Name_List() and "reuse" in inspect.getargspec(tensor_Func).args:                        
                input_Dict.update({
                    "name": '{}'.format(self.reuse_Reference_Dict[reuse_Tensor_Name[0]][reuse_Tensor_Name[1]][1]),
                    "reuse": True   #I don't use AUTO_REUSE yet.
                    });
            else:
                input_Dict.update({"name": name});
                        
            with tf.variable_scope(self.scope_Dict[reuse_Tensor_Name[0] if 'reuse' in input_Dict.keys() else process_Name]) as scope:
                with tf.name_scope(scope.original_name_scope):
                    exported_Tensor = tensor_Func(
                        input_Tensor,
                        **input_Dict,
                        **kwargs
                        )
                
            if "reuse" in inspect.getargspec(tensor_Func).args:
                if not reuse_Tensor_Name is None:
                    self.reuse_Reference_Dict[process_Name][name] = self.reuse_Reference_Dict[reuse_Tensor_Name[0]][reuse_Tensor_Name[1]];
                else:
                    self.reuse_Reference_Dict[process_Name][name] = (process_Name, name);
                
            if type(exported_Tensor) == list:
                for tensor in exported_Tensor:
                    tensor_Name = tensor.name.replace('{}/'.format(process_Name), '');                        
                    if len(set(tensor.name.replace('{}/'.format(process_Name), '').split(':')[0] for tensor in exported_Tensor)) == len(exported_Tensor):
                        tensor_Name = tensor_Name.split(':')[0];
                    self.tensor_Dict[process_Name][tensor_Name] = tensor;
                    self.tensor_Dependency_Dict[process_Name][tensor_Name] = [];
                    for input_Tensor_Name in input_Tensor_Name_List:
                        self.tensor_Dependency_Dict[process_Name][tensor_Name].extend(self.tensor_Dependency_Dict[process_Name][input_Tensor_Name])
                        self.tensor_Connection_Dict[process_Name].append((input_Tensor_Name, tensor_Name));
                    self.tensor_Dependency_Dict[process_Name][tensor_Name] = list(set(self.tensor_Dependency_Dict[process_Name][tensor_Name]));
            else:
                self.tensor_Dict[process_Name][name] = exported_Tensor;
                self.tensor_Dependency_Dict[process_Name][name] = [];
                for input_Tensor_Name in input_Tensor_Name_List:
                    self.tensor_Dependency_Dict[process_Name][name].extend(self.tensor_Dependency_Dict[process_Name][input_Tensor_Name])
                    self.tensor_Connection_Dict[process_Name].append((input_Tensor_Name, name));
                self.tensor_Dependency_Dict[process_Name][name] = list(set(self.tensor_Dependency_Dict[process_Name][name]));
                            
            self.graph_History.append((Tensor_Type.Tensor, {**{"process_Name": process_Name, "name": name, "input_Tensor_Names": input_Tensor_Names, "tensor_Func": tensor_Func, "reuse_Tensor_Name": reuse_Tensor_Name}, **kwargs}));

            if type(exported_Tensor) == list:
                tensor_Name_List = [
                    tensor.name.replace('{}/'.format(process_Name), '').split(':')[0] if len(set(tensor.name.replace('{}/'.format(process_Name), '').split(':')[0] for tensor in exported_Tensor)) == len(exported_Tensor) else tensor.name.replace('{}/'.format(process_Name), '')
                    for tensor in exported_Tensor
                    ]
                return True, [self.Get_Tensor_Shape(process_Name, tensor_Name) for tensor_Name in tensor_Name_List];
            else:
                return True, self.Get_Tensor_Shape(process_Name, name);
        except ValueError as e:            
            return False, e;
        except TypeError as e:            
            return False, e;

    @Check_Lock
    def RNN_Tensor_Generate(self, process_Name, name, input_Tensor_Name, rnn_Cell_Func, state_Func= None, state_Reset= True, reuse_Tensor_Name=None, **kwargs):
        with tf.variable_scope(self.scope_Dict[process_Name]) as scope:
            with tf.name_scope(scope.original_name_scope):
                try:
                    if not input_Tensor_Name in self.Get_Tensor_Name_List(process_Name):
                        raise ValueError("There is no tensor with named '{0}' in process '{1}'.".format(input_Tensor_Name, process_Name))
                    elif name in self.Get_Tensor_Name_List(process_Name):
                        raise ValueError("There is already tensor with named '{0}' in process '{1}'.".format(name, process_Name))
                    elif not reuse_Tensor_Name is None and not reuse_Tensor_Name[1] in self.reuse_Reference_Dict[reuse_Tensor_Name[0]].keys():
                        raise ValueError("Tensor '{0}' in process '{1}' does not support 'reuse'.".format(*reuse_Tensor_Name))
            
                    input_Dict = {};            
                
                    if not reuse_Tensor_Name is None and reuse_Tensor_Name in self.Get_Tensor_Name_List() and "reuse" in inspect.getargspec(rnn_Cell_Func).args:
                        input_Dict.update({
                            "name": self.reuse_Reference_Dict[reuse_Tensor_Name[0]][reuse_Tensor_Name[1]][1],
                            "reuse": True   #I don't use AUTO_REUSE yet.
                            });
                    else:
                        input_Dict.update({"name": name});
                    self.rnn_Cell_Dict[process_Name][name] = rnn_Cell_Func(**input_Dict, **kwargs);            

                    batch_Size = tf.shape(self.tensor_Dict[process_Name][input_Tensor_Name])[0];

                    zero_State = self.rnn_Cell_Dict[process_Name][name].zero_state(batch_size=batch_Size, dtype=tf.float32);
            
                    if state_Reset:
                        initial_State = zero_State;
                    else:
                        if state_Func is None:
                            state_Variable = tf.Variable(initial_value= tf.zeros(zero_State.get_shape().as_list()[1:]), trainable= False)
                            initial_State = tf.tile(tf.expand_dims(state_Variable, axis=0), (batch_Size, 1));
                        else:
                            state_Variable = state_Func(**{
                                state_Name: tf.Variable(initial_value= tf.zeros(state.get_shape().as_list()[1:]), trainable= False)
                                for state_Name, state in zero_State._asdict().items()
                                })
                            initial_State = state_Func(**{
                                state_Name: tf.tile(tf.expand_dims(state, axis=0), (batch_Size, 1))
                                for state_Name, state in state_Variable._asdict().items()
                                })
            
                    self.tensor_Dict[process_Name][name], new_State = tf.nn.dynamic_rnn(
                        inputs= self.tensor_Dict[process_Name][input_Tensor_Name],
                        cell= self.rnn_Cell_Dict[process_Name][name],
                        initial_state= tf.cond(self.is_Training_Placeholder, lambda: initial_State, lambda: zero_State),
                        dtype=tf.float32
                        );
            
                    if not state_Reset:
                        if state_Func is None:
                            self.rnn_State_Assign_List.append(tf.assign(state_Variable, new_State[0]));
                        else:
                            for state_Name in new_State._asdict().keys():                        
                                self.rnn_State_Assign_List.append(tf.assign(state_Variable._asdict()[state_Name], new_State._asdict()[state_Name][0]));

                    if not reuse_Tensor_Name is None and "reuse" in inspect.getargspec(rnn_Cell_Func).args and \
                        reuse_Tensor_Name[1] in self.reuse_Reference_Dict[reuse_Tensor_Name[0]].keys():
                        self.reuse_Reference_Dict[process_Name][name] = self.reuse_Reference_Dict[reuse_Tensor_Name[0]][reuse_Tensor_Name[1]];
                    else:
                        self.reuse_Reference_Dict[process_Name][name] = (process_Name, name);

                    self.graph_History.append((Tensor_Type.RNN_Tensor, {**{"process_Name": process_Name, "name": name, "input_Tensor_Name": input_Tensor_Name, "rnn_Cell_Func": rnn_Cell_Func, "state_Func": state_Func, "state_Reset": state_Reset}, **kwargs}));
                    self.tensor_Dependency_Dict[process_Name][name] = self.tensor_Dependency_Dict[process_Name][input_Tensor_Name];
                    self.tensor_Connection_Dict[process_Name].append((input_Tensor_Name, name))

                    return True, self.Get_Tensor_Shape(process_Name, name);

                except ValueError as e:
                    return False, e;
                except TypeError as e:            
                    return False, e;
        
    @Check_Lock
    def Create_Tensor_Generate(self, process_Name, name, tensor_Func, shape, **kwargs):        
        try:
            if name in self.Get_Tensor_Name_List(process_Name):
                raise ValueError("There is already tensor with named '{0}' in process '{1}'.".format(name, process_Name))            
            elif len(self.placeholder_Dict[process_Name]) == 0:
                raise ValueError("There is no placeholder in process '{0}'. At least one placeholder must exist in the process for the batch size reference.".format(name, process_Name))            
                               
            with tf.variable_scope(self.scope_Dict[process_Name]) as scope:
                with tf.name_scope(scope.original_name_scope):
                    exported_Tensor = tensor_Func(
                        name= name,
                        shape= [tf.shape([x for x in self.placeholder_Dict[process_Name].values()][0])[0]] + shape,
                        **kwargs
                        )
            
            if type(exported_Tensor) == list:
                for tensor in exported_Tensor:
                    tensor_Name = tensor.name.replace('{}/'.format(process_Name), '');                    
                    if len(set(tensor.name.replace('{}/'.format(process_Name), '').split(':')[0] for tensor in exported_Tensor)) == len(exported_Tensor):
                        tensor_Name = tensor_Name.split(':')[0];                    
                    self.tensor_Dict[process_Name][tensor_Name] = tensor;

                    self.tensor_Dependency_Dict[process_Name][tensor_Name] = [];    #No dependency
                    self.tensor_Connection_Dict[process_Name].append((tensor_Name, tensor_Name));
            else:
                self.tensor_Dict[process_Name][name] = exported_Tensor;                
                self.create_Dict[process_Name][name] = exported_Tensor;
                self.tensor_Dependency_Dict[process_Name][name] = [];       #No dependency
                self.tensor_Connection_Dict[process_Name].append((name, name));                
                
            self.graph_History.append((Tensor_Type.Create, {**{"process_Name": process_Name, "name": name, "tensor_Func": tensor_Func, "shape": shape}, **kwargs}));

            if type(exported_Tensor) == list:
                tensor_Name_List = [
                    tensor.name.replace('{}/'.format(process_Name), '').split(':')[0] if len(set(tensor.name.replace('{}/'.format(process_Name), '').split(':')[0] for tensor in exported_Tensor)) == len(exported_Tensor) else tensor.name
                    for tensor in exported_Tensor
                    ]
                return True, [self.Get_Tensor_Shape(process_Name, tensor_Name) for tensor_Name in tensor_Name_List];
            else:
                return True, self.Get_Tensor_Shape(process_Name, name);

        except ValueError as e:            
            return False, e;
        except TypeError as e:            
            return False, e;

    @Check_Lock
    @Check_Optimizer
    def Loss_Generate(self, process_Name, name, label_Tensor_Name, loss_Func, logit_Tensor_Name=None, prediction_Tensor_Name=None, weights = 1):        
        try:
            if name in self.Get_Tensor_Name_List(process_Name):
                raise ValueError("There is already tensor with named '{0}' in process '{1}'.".format(name, process_Name))            
            elif not label_Tensor_Name in self.Get_Tensor_Name_List(process_Name):
                raise ValueError("There is no tensor with named '{0}' in process '{1}'.".format(label_Tensor_Name, process_Name))
            elif (logit_Tensor_Name is None and prediction_Tensor_Name is None) or (not logit_Tensor_Name is None and not prediction_Tensor_Name is None):
                raise ValueError("One of 'logit_Tensor_Name' or 'prediction_Tensor_Name' parameter must be assigned.".format(logit_Tensor_Name, process_Name))            
            elif not logit_Tensor_Name is None and not logit_Tensor_Name in self.Get_Tensor_Name_List(process_Name):
                raise ValueError("There is no tensor with named '{0}' in process '{1}'.".format(logit_Tensor_Name, process_Name))            
            elif not prediction_Tensor_Name is None and not prediction_Tensor_Name in self.Get_Tensor_Name_List(process_Name):
                raise ValueError("There is no tensor with named '{0}' in process '{1}'.".format(prediction_Tensor_Name, process_Name))            
                            
            with tf.variable_scope(self.scope_Dict[process_Name]) as scope:
                with tf.name_scope(scope.original_name_scope):
                    new_Loss = tf.identity(
                        input= loss_Func(
                            self.tensor_Dict[process_Name][label_Tensor_Name],
                            self.tensor_Dict[process_Name][logit_Tensor_Name or prediction_Tensor_Name],
                            weights = weights or 1
                            ),
                        name= name
                        )

            self.tensor_Dict[process_Name][name] = new_Loss;
            self.loss_Dict[process_Name][name] = new_Loss;

            self.graph_History.append((Tensor_Type.Loss, {"process_Name": process_Name, "name": name, "label_Tensor_Name": label_Tensor_Name, "loss_Func": loss_Func, "logit_Tensor_Name": logit_Tensor_Name, "prediction_Tensor_Name": prediction_Tensor_Name, "weights": weights}));
            self.tensor_Dependency_Dict[process_Name][name] = list(set(self.tensor_Dependency_Dict[process_Name][label_Tensor_Name] + self.tensor_Dependency_Dict[process_Name][logit_Tensor_Name or prediction_Tensor_Name]));

            #This will be changed by the library.
            self.tensor_Connection_Dict[process_Name].append((label_Tensor_Name, name));
            self.tensor_Connection_Dict[process_Name].append((logit_Tensor_Name or prediction_Tensor_Name, name));

            return True, self.Get_Tensor_Shape(process_Name, name);

        except ValueError as e:
            return False, e;
        except TypeError as e:            
            return False, e;

    @Check_Lock
    @Check_Optimizer
    def Optimizer_Generate(self, process_Name, optimizer_Func, **kwargs):
        '''
        applied_loss
        applied_variables
        initial_learning_rate
        decay_method
        decay_steps
        decay_rate
        warmup_steps
        '''        
        try:
            applied_loss = list(self.loss_Dict[process_Name].keys());
            initial_learning_rate = 0.001;
            decay_method = Learning_Rate_Decay_Method.No_Decay;
            applied_variable = tf.trainable_variables()
            
            if 'applied_loss' in kwargs.keys() and not kwargs['applied_loss'] is None:
                applied_loss = kwargs['applied_loss']               
            if 'initial_learning_rate' in kwargs.keys() and not kwargs['initial_learning_rate'] is None:
                initial_learning_rate = kwargs['initial_learning_rate']                
            if 'decay_method' in kwargs.keys() and not kwargs['decay_method'] is None:
                decay_method = kwargs['decay_method']
            if 'applied_variable' in kwargs.keys() and not kwargs['applied_variable'] is None and len(kwargs['applied_variable']) > 0:
                applied_variable = [x for x in tf.trainable_variables() if x.name in kwargs['applied_variable']];
                
            self.tensor_Dict[process_Name]["{}.Loss".format(process_Name)] = tf.add_n(
                [tensor for loss_Name, tensor in self.loss_Dict[process_Name].items() if loss_Name in applied_loss],
                name= "{}.Loss".format(process_Name)
                );
            
            if decay_method == Learning_Rate_Decay_Method.No_Decay:
                learning_Rate = tf.cast(initial_learning_rate, tf.float32);
            elif decay_method == Learning_Rate_Decay_Method.Exponential:
                decay_steps = kwargs['decay_steps'] if 'decay_steps' in kwargs.keys() and not kwargs['decay_steps'] is None else 10000;
                decay_rate = kwargs['decay_rate'] if 'decay_rate' in kwargs.keys() and not kwargs['decay_rate'] is None else 0.5;
                learning_Rate = tf.train.exponential_decay(
                    learning_rate=initial_learning_rate,
                    global_step=self.global_Tensor_Dict['Variable_Dict']['global_Step'],
                    decay_steps= decay_steps,
                    decay_rate= decay_rate
                    )
            elif decay_method == Learning_Rate_Decay_Method.Noam:
                warmup_steps= kwargs['warmup_steps'] if 'warmup_steps' in kwargs.keys() and not kwargs['warmup_steps'] is None else 4000;
                step = tf.cast(self.global_Tensor_Dict['Variable_Dict']["global_Step"] + 1, dtype=tf.float32);
                learning_Rate = initial_learning_rate * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps**-1.5, step**-0.5);
                        
            with tf.variable_scope(self.scope_Dict[process_Name]) as scope: #This may be not meaning.
                with tf.name_scope(scope.original_name_scope):
                    optimizer = optimizer_Func(
                        learning_rate= learning_Rate,
                        name= "{}.Optimizer".format(process_Name),                
                        **{
                            parameter: kwargs[parameter]
                            for parameter in inspect.getargspec(optimizer_Func).args
                            if parameter in kwargs.keys() and parameter != 'self' and kwargs[parameter] is not None
                            }
                        )
            
                    gradients, variables = zip(*[
                        (gradient, variable)
                        for gradient, variable in optimizer.compute_gradients(self.tensor_Dict[process_Name]["{}.Loss".format(process_Name)])
                        if variable in applied_variable
                        ])
                    clipped_Gradients, global_Norm = tf.clip_by_global_norm(gradients, 1.0)
            
                    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                        self.tensor_Dict[process_Name]["{}.Optimizer".format(process_Name)] = optimizer.apply_gradients(
                            zip(clipped_Gradients, variables),
                            name= "{}.Optimizer".format(process_Name)
                            )
            
            self.graph_History.append((Tensor_Type.Optimizer, {**{"process_Name": process_Name, "optimizer_Func": optimizer_Func}, **kwargs}));
            
            self.tensor_Dependency_Dict[process_Name]["{}.Loss".format(process_Name)] = [];
            self.tensor_Dependency_Dict[process_Name]["{}.Optimizer".format(process_Name)] = [];
            for loss_Name in [loss_Name for loss_Name, tensor in self.loss_Dict[process_Name].items() if loss_Name in applied_loss]:
                self.tensor_Dependency_Dict[process_Name]["{}.Loss".format(process_Name)].extend(self.tensor_Dependency_Dict[process_Name][loss_Name]);
                self.tensor_Dependency_Dict[process_Name]["{}.Optimizer".format(process_Name)].extend(self.tensor_Dependency_Dict[process_Name][loss_Name]);

            self.tensor_Dependency_Dict[process_Name]["{}.Loss".format(process_Name)] = list(set(self.tensor_Dependency_Dict[process_Name]["{}.Loss".format(process_Name)]));
            self.tensor_Dependency_Dict[process_Name]["{}.Optimizer".format(process_Name)] = list(set(self.tensor_Dependency_Dict[process_Name]["{}.Optimizer".format(process_Name)]));

            #This will be changed by the library.
            for loss_Name, _ in self.loss_Dict[process_Name].items():
                if loss_Name in applied_loss:
                    self.tensor_Connection_Dict[process_Name].append((loss_Name, "{}.Optimizer".format(process_Name)));
            
            return True, None;
        except ValueError as e:
            return False, e;
        except TypeError as e:            
            return False, e;

    @Check_Lock
    def Tensor_Initialize(self):
        self.tf_Session.run(tf.global_variables_initializer());
        self.is_Locked = True;

    def Get_Tensor_Shape(self, process_Name, name):
        if type(self.tensor_Dict[process_Name][name]) == tf.Operation:
            return None;

        return self.tensor_Dict[process_Name][name].get_shape().as_list()[1:]

    def Get_Tensor_Dependency(self, process_Name, name):
        return self.tensor_Dependency_Dict[process_Name][name];

    def Get_Variable_List(self):
        return [v.name for v in tf.trainable_variables()];

    def Process_Save(self, save_Path):
        save_Dict = {
            "Process_Name_List": list(self.tensor_Dict.keys()),
            "Graph_History": self.graph_History
            }

        with open(save_Path, "wb") as f:
            pickle.dump(save_Dict, f, protocol= 2);

    def Process_Load(self, load_Path):
        with open(load_Path, "rb") as f:            
            load_Dict = pickle.load(f);

        self._Process_Load(load_Dict["Process_Name_List"], load_Dict["Graph_History"]);

    def _Process_Load(self, process_Name_List, graph_History):            
        for process_Name in process_Name_List:
            self.Set_New_Process(process_Name);
            
        for tensor_Type, tensor_Information in graph_History:
            if tensor_Type == Tensor_Type.Placeholder:
                self.Placeholder_Generate(**tensor_Information);
            if tensor_Type == Tensor_Type.Tensor:
                self.Tensor_Generate(**tensor_Information);
            if tensor_Type == Tensor_Type.RNN_Tensor:
                self.RNN_Tensor_Generate(**tensor_Information);
            if tensor_Type == Tensor_Type.Create:
                self.Create_Tensor_Generate(**tensor_Information);
            if tensor_Type == Tensor_Type.Loss:
                self.Loss_Generate(**tensor_Information);
            if tensor_Type == Tensor_Type.Optimizer:
                self.Optimizer_Generate(**tensor_Information);

    def Network_Graph(self, process_Name):
        nx_Graph = nx.DiGraph();
        color_List = [];
        label_Dict = {};

        for tensor_Name in set([tensor_Name for connection in self.tensor_Connection_Dict[process_Name] for tensor_Name in connection]):
            nx_Graph.add_node(tensor_Name);
            
            if tensor_Name in self.placeholder_Dict[process_Name].keys():
                color_List.append('#F8766D')            
            elif tensor_Name in self.loss_Dict[process_Name].keys():
                color_List.append('#7CAE00')
            elif ".Optimizer" in tensor_Name:
                color_List.append('#C77CFF')
            else:
                color_List.append('#00BFC4')

            new_Tag_List = [tensor_Name];
            if tensor_Name in self.reuse_Reference_Dict[process_Name].keys() and (process_Name, tensor_Name) != self.reuse_Reference_Dict[process_Name][tensor_Name]:
                if self.reuse_Reference_Dict[process_Name][tensor_Name][0] == process_Name:
                    new_Tag_List.append('({})'.format(self.reuse_Reference_Dict[process_Name][tensor_Name][1]))
                else:
                    new_Tag_List.append('{}'.format(self.reuse_Reference_Dict[process_Name][tensor_Name]).replace('\'', ''))
            if not ".Optimizer" in tensor_Name and len(self.tensor_Dict[process_Name][tensor_Name].shape.as_list()) > 0:
                new_Tag_List.append('{}'.format(['B'] + self.tensor_Dict[process_Name][tensor_Name].shape.as_list()[1:]).replace('\'', ''))
            label_Dict[tensor_Name] = "\n".join(new_Tag_List)

        for from_Tensor, to_Tensor in self.tensor_Connection_Dict[process_Name]:
            nx_Graph.add_edge(from_Tensor, to_Tensor, weight=1)
            
        return nx_Graph, label_Dict, color_List;