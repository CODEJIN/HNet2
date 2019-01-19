import numpy as np;
import tensorflow as tf;
import os, sys, inspect, collections;
import _pickle as pickle;

class Learning_Manager:
    def Check_Lock(func):
        def wrapper(self, *args, **kwargs):
            if self.is_Locked:
                raise Exception("Learning modifying is locked.")
            return func(self, *args, **kwargs)
        return wrapper

    def __init__(self, process_Manager, pattern_Manager):
        self.process_Manager = process_Manager;
        self.pattern_Manager = pattern_Manager;

        self.learning_List = [];    #Learning needs sequence, ordereddict doesn't support exchange.
        self.test_Dict = {};    #Key is used for result identifier
        
        self.is_Locked = False;

    def Get_Matching_Info(self, process_Name, pattern_Name, matching_List, probability_Pattern_Column_Name= None, get_Tensor_Name_List= None):        
        '''
        Matching_Info: A dict. {"Process": process_Name, "Pattern": pattern_Name, "Matching_List": tuple list of (placeholder_Name, pattern_Column_Name)}
        '''
        try:
            if get_Tensor_Name_List is None:    #for training
                dependency_Placeholder_Name_List = self.process_Manager.Get_Tensor_Dependency(process_Name, "{}.Optimizer".format(process_Name));
            else:
                dependency_Placeholder_Name_List = [];
                for tensor_Name in get_Tensor_Name_List:
                    dependency_Placeholder_Name_List.extend(self.process_Manager.Get_Tensor_Dependency(process_Name, tensor_Name));
                dependency_Placeholder_Name_List = list(set(dependency_Placeholder_Name_List));                

            pattern_Column_Name_List = list(self.pattern_Manager.pattern_Dict[pattern_Name]);

            assigned_Placeholder_Name_List = [x for x, _ in matching_List]
            assigned_Pattern_Column_Name_List = [x for _, x in matching_List]

            if not sorted(dependency_Placeholder_Name_List) == sorted(assigned_Placeholder_Name_List):
                raise ValueError("The assigned placeholders are incompatible with the dependency of process '{0}'.".format(process_Name));
            for assigned_Placeholder_Name in assigned_Placeholder_Name_List:
                if not assigned_Placeholder_Name in self.process_Manager.placeholder_Dict[process_Name].keys():
                    raise ValueError("There is no placeholder '{0}' in process '{1}'.".format(assigned_Placeholder_Name, process_Name));
            for assigned_Pattern_Column_Name in assigned_Pattern_Column_Name_List:
                if not assigned_Pattern_Column_Name in pattern_Column_Name_List:
                    raise ValueError("There is no pattern column '{0}' in pattern '{1}'.".format(assigned_Pattern_Column_Name, pattern_Name));
            if not probability_Pattern_Column_Name is None:
                if not probability_Pattern_Column_Name in pattern_Column_Name_List:
                    raise ValueError("There is no pattern column '{0}' in pattern '{1}'.".format(probability_Pattern_Column_Name, pattern_Name));
                elif self.pattern_Manager.Get_Pattern_Shape(pattern_Name, probability_Pattern_Column_Name) != ():
                    raise ValueError("Pattern column '{0}' is not single numeric'.".format(probability_Pattern_Column_Name));

            for matching in matching_List:
                placeholder_Shape = self.process_Manager.Get_Tensor_Shape(process_Name, matching[0])
                pattern_Shape = self.pattern_Manager.Get_Pattern_Shape(pattern_Name, matching[1])
                if not all([x ==  y for x, y in zip(placeholder_Shape, pattern_Shape)] + [len(placeholder_Shape) == len(pattern_Shape)]):
                    raise ValueError("The shapes of placeholder '{0}' and pattern column '{1}' do not matched.".format(matching[0], matching[1]));
            
            new_Matching_Info = {
                "Process": process_Name,
                "Pattern": pattern_Name,
                "Matching_List": matching_List,
                'Probability': probability_Pattern_Column_Name
                }
            
            return True, new_Matching_Info;

        except ValueError as e:
            return False, e;
        
    @Check_Lock
    def Set_New_Learning(self, learning_Name, epoch, checkpoint_Save_Interval, test_Interval, mini_Batch, use_Probability):
        try:
            if any([learning_Name == learning['Name'] for learning in self.learning_List]):
                raise ValueError("There is already a learning infomation with named '{}'.".format(learning_Name))

            self.learning_List.append({
                'Name': learning_Name,
                'Epoch': epoch,
                'Checkpoint_Save_Interval': checkpoint_Save_Interval,
                'Test_Interval': test_Interval,            
                'Mini_Batch': mini_Batch,
                'Matching_Info_List': [],
                'Use_Probability': use_Probability
                })

            return True, None;
        except ValueError as e:
            return False, e;

    @Check_Lock    
    def Learning_Add_Mathing_Info(self, learning_Name, matching_Info):
        try:
            learning_Info = [x for x in self.learning_List if learning_Name == x['Name']][0]
            if learning_Info['Use_Probability']:            
                if matching_Info['Probability'] is None:
                    raise ValueError("Matching info must have correct probability pattern column name when 'use_Probability' is 'True'.");
            learning_Info['Matching_Info_List'].append(matching_Info);

            return True, None;
        except ValueError as e:
            return False, e;

    @Check_Lock
    def Set_New_Test(self, test_Name, mini_Batch, matching_Info, get_Tensor_Name_List, attached_Pattern_Column_Name_List):
        try:
            if test_Name in self.test_Dict.keys():
                raise ValueError("There is already test with named '{0}'.".format(test_Name))
            dependency_Placeholder_Name_List = [];
            for tensor_Name in get_Tensor_Name_List:
                dependency_Placeholder_Name_List.extend(self.process_Manager.Get_Tensor_Dependency(matching_Info["Process"], tensor_Name));
            dependency_Placeholder_Name_List = list(set(dependency_Placeholder_Name_List));
            assigned_Placeholder_Name_List = [x for x, _ in matching_Info["Matching_List"]];
            if not sorted(dependency_Placeholder_Name_List) == sorted(assigned_Placeholder_Name_List):
                raise ValueError("The assigned placeholders are incompatible with the dependency of tensors to aquire.");
            for pattern_Column_Name in attached_Pattern_Column_Name_List:
                if not pattern_Column_Name in self.pattern_Manager.pattern_Dict[matching_Info['Pattern']]:
                    raise ValueError("There is no pattern column with named '{0}' in pattern '{1}'.".format(pattern_Column_Name, matching_Info['Pattern']));
            self.test_Dict[test_Name] = {            
                'Mini_Batch': mini_Batch,
                'Matching_Info': matching_Info,
                'Get_Tensor_Name_List': get_Tensor_Name_List,
                'Attached_Pattern_Column_Name_List': attached_Pattern_Column_Name_List
                }
            return True, None;

        except ValueError as e:
            return False, e;
    
    def Lock(self):
        self.is_Locked = True;

    def Learning_Save(self, save_Path):
        save_Dict = {
            "Learning_List": self.learning_List,
            "Test_Dict": self.test_Dict
            }

        with open(save_Path, "wb") as f:
            pickle.dump(save_Dict, f, protocol= 2);

    @Check_Lock
    def Learning_Load(self, load_Path):
        with open(load_Path, "rb") as f:            
            load_Dict = pickle.load(f);

        self._Process_Load(load_Dict["Learning_List"], load_Dict["Test_Dict"]);

    @Check_Lock
    def _Learning_Load(self, learning_List, test_Dict):
        try:
            for learning_Info in learning_List:
                self.Set_New_Learning(
                    learning_Name= learning_Info['Name'],
                    epoch= learning_Info['Epoch'],
                    checkpoint_Save_Interval= learning_Info['Checkpoint_Save_Interval'],
                    test_Interval= learning_Info['Test_Interval'],
                    mini_Batch= learning_Info['Mini_Batch'],
                    use_Probability= learning_Info['Use_Probability']
                    )
                
                for matching_Info in learning_Info['Matching_Info_List']:                    
                    response, new_Matching_Info = self.Get_Matching_Info(
                        process_Name= matching_Info['Process'],
                        pattern_Name= matching_Info['Pattern'],
                        matching_List= matching_Info['Matching_List'],
                        probability_Pattern_Column_Name= matching_Info['Probability']
                        )
                    if not response:
                        return False, new_Matching_Info;

                    self.Learning_Add_Mathing_Info(learning_Info['Name'], new_Matching_Info);
                    
            for test_Name, test_Info in test_Dict.items():
                response, new_Matching_Info = self.Get_Matching_Info(
                    process_Name= test_Info['Matching_Info']['Process'],
                    pattern_Name= test_Info['Matching_Info']['Pattern'],
                    matching_List= test_Info['Matching_Info']['Matching_List'],
                    probability_Pattern_Column_Name= test_Info['Matching_Info']['Probability'],
                    get_Tensor_Name_List= test_Info['Get_Tensor_Name_List']
                    )
                if not response:
                    return False, new_Matching_Info;

                self.Set_New_Test(
                    test_Name= test_Name,
                    mini_Batch= test_Info['Mini_Batch'],
                    matching_Info= new_Matching_Info,
                    get_Tensor_Name_List= test_Info['Get_Tensor_Name_List'],
                    attached_Pattern_Column_Name_List= test_Info['Attached_Pattern_Column_Name_List']
                    )

        except ValueError as e:
            return False, e;

        