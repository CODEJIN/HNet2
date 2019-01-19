import numpy as np;
import tensorflow as tf;
import pandas as pd;
import os, sys, time;
import _pickle as pickle;
from threading import Thread;
from collections import deque;
from random import shuffle;

class Pattern_Manager:
    def __init__(self, process_Manager, max_Queue = 10):
        self.process_Manager = process_Manager;
        self.max_Queue = max_Queue;

        self.pattern_Dict = {};

        self.training_Pattern_Queue = deque();
        self.test_Pattern_Dict = {};

        self.is_Pattern_Generate = False;
        self.is_Learning_Finished = False;
        self.is_Test_Pattern_Generate_Finished = False;
            
    def Pattern_Load(self, pattern_Name, pattern_Path):
        try:
            if pattern_Name in self.pattern_Dict.keys():
                raise ValueError("There is already pattern with named '{0}'.".format(pattern_Name))

            with open(pattern_Path, "rb") as f:
                self.pattern_Dict[pattern_Name] = pickle.load(f);   #pattern is pd.DataFrame

            return True, None;

        except ValueError as e:            
            return False, e;
                
    def Pattern_Generate(
        self,
        learning_Manager,
        global_Step= 0,
        global_Epoch= 0,
        local_Step= 0,
        local_Epoch= 0,
        learning_Flow_Index= 0
        ):
        try:
            if not learning_Manager.is_Locked:
                raise ValueError("Learning manager should be locked.");
            self.learning_Manager = learning_Manager;
            
            training_Pattern_Generate_Thread = Thread(
                target=self.Training_Pattern_Generate,
                args=(global_Step, global_Epoch, local_Step, local_Epoch, learning_Flow_Index)
                )
            training_Pattern_Generate_Thread.daemon = True;
            training_Pattern_Generate_Thread.start();
            
            test_Pattern_Generate_Thread = Thread(target=self.Test_Pattern_Generate);
            test_Pattern_Generate_Thread.daemon = True;
            test_Pattern_Generate_Thread.start();
            
            self.is_Pattern_Generate = True;

            return True, None
        except ValueError as e:
            return False, e
            
    def Training_Pattern_Generate(
        self,
        start_Global_Step= 0,
        start_Global_Epoch= 0,
        start_Local_Step= 0,
        start_Local_Epoch= 0,
        start_Learning_Flow_Index= 0
        ):
        global_Step= start_Global_Step;
        global_Epoch = start_Global_Epoch;
        local_Step = start_Local_Step;

        for learning_Flow_Index, learning_Info in enumerate(self.learning_Manager.learning_List[start_Learning_Flow_Index:], start_Learning_Flow_Index):
            for local_Epoch in range(start_Local_Epoch, learning_Info["Epoch"]):                
                start_Local_Epoch = 0 #Only current learning should be started from 'start epoch' parameter.
                is_Checkpoint_Save_Interval = (local_Epoch % learning_Info['Checkpoint_Save_Interval'] == 0);
                is_Test_Interval = (local_Epoch % learning_Info['Test_Interval'] == 0);

                training_Batch_List = [];   #(process_Name, pattern_Name, matching, batch_Index_List)
                for matching_Info in learning_Info["Matching_Info_List"]:
                    process_Name = matching_Info["Process"];
                    pattern_Name = matching_Info["Pattern"];
                    matching_List = matching_Info["Matching_List"];
                    probability_Pattern_Column_Name = matching_Info["Probability"];

                    pattern_Index_List = list(range(len(self.pattern_Dict[pattern_Name])));

                    if learning_Info["Use_Probability"]:                        
                        probability_Determination_List = self.pattern_Dict[pattern_Name][probability_Pattern_Column_Name].tolist() < np.random.random(len(self.pattern_Dict[pattern_Name]));
                        pattern_Index_List = [index for index, determination in zip(pattern_Index_List, probability_Determination_List) if determination];

                    shuffle(pattern_Index_List);
                    training_Batch_List.extend([(process_Name, pattern_Name, matching_List, pattern_Index_List[x:x+learning_Info["Mini_Batch"]]) for x in range(0, len(pattern_Index_List), learning_Info["Mini_Batch"])]);
                shuffle(training_Batch_List)

                current_Index = 0;                
                while current_Index < len(training_Batch_List):
                    if len(self.training_Pattern_Queue) >= self.max_Queue:
                        time.sleep(0.1);
                        continue;

                    is_End_of_Epoch = current_Index + 1 >= len(training_Batch_List);

                    process_Name, pattern_Name, matching_List, batch_Index_List = training_Batch_List[current_Index];
                    batch_Pattern_Dataframe = self.pattern_Dict[pattern_Name].loc[batch_Index_List]    #batch patterns dataframe
                                        
                    new_Fetches = [
                        self.process_Manager.global_Tensor_Dict['Assign_Tensor_Dict']['global_Step'],
                        self.process_Manager.global_Tensor_Dict['Assign_Tensor_Dict']['global_Epoch'],
                        self.process_Manager.global_Tensor_Dict['Assign_Tensor_Dict']['learning_Flow_Index'],
                        self.process_Manager.global_Tensor_Dict['Assign_Tensor_Dict']['local_Step'],
                        self.process_Manager.global_Tensor_Dict['Assign_Tensor_Dict']['local_Epoch'],
                        self.process_Manager.tensor_Dict[process_Name]["{}.Loss".format(process_Name)],
                        self.process_Manager.tensor_Dict[process_Name]["{}.Optimizer".format(process_Name)]
                        ]
                    new_Fetches.append(self.process_Manager.rnn_State_Assign_List);
                    new_Feed_Dict = {
                        self.process_Manager.global_Tensor_Dict['Placeholder_Dict']['global_Step']: global_Step + 1,
                        self.process_Manager.global_Tensor_Dict['Placeholder_Dict']['global_Epoch']: global_Epoch + 1,
                        self.process_Manager.global_Tensor_Dict['Placeholder_Dict']['learning_Flow_Index']: learning_Flow_Index,
                        self.process_Manager.global_Tensor_Dict['Placeholder_Dict']['local_Step']: local_Step + 1,
                        self.process_Manager.global_Tensor_Dict['Placeholder_Dict']['local_Epoch']: local_Epoch + 1
                        }
                    new_Feed_Dict.update({                        
                        self.process_Manager.placeholder_Dict[process_Name][placeholder_Name]: np.stack(batch_Pattern_Dataframe[pattern_Column_Name].values)
                        for placeholder_Name, pattern_Column_Name in matching_List
                        })
                    new_Feed_Dict[self.process_Manager.is_Training_Placeholder] = True;

                    self.training_Pattern_Queue.append([
                        is_Checkpoint_Save_Interval,
                        is_Test_Interval,
                        is_End_of_Epoch,
                        process_Name,
                        pattern_Name,
                        new_Fetches,
                        new_Feed_Dict
                        ])
                
                    current_Index += 1;
                    global_Step += 1;
                    local_Step += 1;
                    is_Checkpoint_Save_Interval = False;
                    is_Test_Interval = False;
                    
                global_Epoch += 1;
        self.is_Learning_Finished = True;
            
    def Test_Pattern_Generate(self):
        for test_Name, test_Info in self.learning_Manager.test_Dict.items():
            self.test_Pattern_Dict[test_Name] = [];

            mini_Batch = test_Info['Mini_Batch'];
            process_Name = test_Info['Matching_Info']["Process"];
            pattern_Name = test_Info['Matching_Info']["Pattern"];
            matching_List = test_Info['Matching_Info']["Matching_List"];
            get_Tensor_Name_List = test_Info["Get_Tensor_Name_List"];
            attached_Pattern_Column_Name_List = test_Info["Attached_Pattern_Column_Name_List"];

            pattern_Index_List = list(range(len(self.pattern_Dict[pattern_Name])));
            for batch_Index_List in [pattern_Index_List[x:x+mini_Batch] for x in range(0, len(pattern_Index_List), mini_Batch)]:
                batch_Pattern_Dataframe = self.pattern_Dict[pattern_Name].loc[batch_Index_List]    #batch patterns dataframe

                new_Fetches = {
                    tensor_Name: self.process_Manager.tensor_Dict[process_Name][tensor_Name]
                    for tensor_Name in get_Tensor_Name_List
                    }

                new_Feed_Dict = {
                    self.process_Manager.placeholder_Dict[process_Name][placeholder_Name]: np.stack(batch_Pattern_Dataframe[pattern_Column_Name].values)
                    for placeholder_Name, pattern_Column_Name in matching_List
                    }
                new_Feed_Dict[self.process_Manager.is_Training_Placeholder] = False;

                new_Attachment_Column_Dict = {
                    pattern_Column_Name: batch_Pattern_Dataframe[pattern_Column_Name].values
                    for pattern_Column_Name in attached_Pattern_Column_Name_List
                    }
                
                self.test_Pattern_Dict[test_Name].append([process_Name, pattern_Name, new_Fetches, new_Feed_Dict, new_Attachment_Column_Dict]);

        self.is_Test_Pattern_Generate_Finished = True;
        
    def Get_Pattern(self):        
        while len(self.training_Pattern_Queue) == 0: #When training speed is faster than making pattern, model should be wait.
            time.sleep(0.01);
        return self.training_Pattern_Queue.popleft();
    
    def Get_Test_Pattern(self):
        while not self.is_Test_Pattern_Generate_Finished: #When model generating speed is faster than making pattern, model should be wait.
            time.sleep(0.1);
        return self.test_Pattern_Dict;

    def Get_Pattern_Shape(self, pattern_Name, pattern_Column_Name):
        try:
            return self.pattern_Dict[pattern_Name][pattern_Column_Name][0].shape  #When a numeric is inserted
        except:
            if type(self.pattern_Dict[pattern_Name][pattern_Column_Name][0]) in [int, float, complex]:
                return ();
            #non-numeric like string
            return False;