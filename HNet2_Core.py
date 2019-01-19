import os, sys, inspect, collections;
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np;
import tensorflow as tf;
import pandas as pd;
import _pickle as pickle;
from threading import Thread;
import gc;
from HNet_Enum import Tensor_Type, Learning_Rate_Decay_Method, Model_State;
from Manager.Process import Process_Manager;
from Manager.Pattern import Pattern_Manager;
from Manager.Learning import Learning_Manager;
import Compatible_Functions;


class HNet:
    def Check_Prerequisite(func):
        def wrapper(self, *args, **kwargs):
            if self.save_Path is None:
                raise Exception("Save path should be assigned before train. Use the function 'Set_Save_Path'.")
            if not self.pattern_Manager.is_Pattern_Generate:
                raise Exception("Pattern is not generating. Prerequisite may not finished yet. Check the commands you set up.")
            if not self.pattern_Manager.is_Pattern_Generate:
                raise Exception("Pattern is not generating. Prerequisite may not finished yet. Check the commands you set up.")
            if self.state == Model_State.Finished:
                raise Exception("Training is already finished.")
            return func(self, *args, **kwargs)
        return wrapper

    def __init__(self):
        self.tf_Session = tf.Session();

        self.process_Manager = Process_Manager(self.tf_Session);
        self.save_Path = None;
        
        self.state = Model_State.Paused;
        self.training_Loss_DataFrame = pd.DataFrame(columns=["Global_Step", "Global_Epoch", "Learning_Flow_Index", "Local_Step", "Local_Epoch", "Process_Name", "Pattern_Name", "Loss"])

    def Set_Save_Path(self, save_Path):
        if not os.path.exists(save_Path):
            os.makedirs(save_Path);

        self.save_Path = save_Path;

    def Run_Pattern_Manager(self):
        try:
            self.tf_Saver = tf.train.Saver(max_to_keep=0);

            if not self.process_Manager.is_Locked:
                raise Exception("Process manager should be locked to run pattern manager.")
            self.pattern_Manager = Pattern_Manager(process_Manager= self.process_Manager);
            return True, None
        except Exception as e:
            return False, e

    def Run_Learning_Manager(self):
        try:
            if not self.process_Manager.is_Locked:
                raise Exception("Process manager should be locked to run pattern manager.")
            self.learning_Manager = Learning_Manager(process_Manager= self.process_Manager, pattern_Manager= self.pattern_Manager);
            return True, None
        except Exception as e:
            return False, e
    
    @Check_Prerequisite
    def Train(self):
        train_Thread = Thread(target=self._Train);        
        train_Thread.daemon = True;
        train_Thread.start();

    @Check_Prerequisite
    def _Train(self):
        self.state = Model_State.Running

        while not self.pattern_Manager.is_Learning_Finished or len(self.pattern_Manager.training_Pattern_Queue) > 0:    #When there is no more training pattern, the train function will be done.            
            is_Checkpoint_Save_Interval, is_Test_Interval, is_End_of_Epoch, process_Name, pattern_Name, fetches, feed_Dict = self.pattern_Manager.Get_Pattern();

            if is_Checkpoint_Save_Interval:
                self.Checkpoint_Save();

            if is_Test_Interval:
                self.Test();  
            
            global_Step, global_Epoch, learning_Flow_Index, local_Step, local_Epoch, loss = self.tf_Session.run(
                fetches = fetches,
                feed_dict = feed_Dict
                )[:6]
            
            self.training_Loss_DataFrame.loc[len(self.training_Loss_DataFrame)] = [
                global_Step, global_Epoch, learning_Flow_Index, local_Step, local_Epoch, process_Name, pattern_Name, loss
                ]

            if self.state == Model_State.Pausing and is_End_of_Epoch:
                self.state = Model_State.Paused;
                break;

        if not self.state == Model_State.Paused:
            self.Checkpoint_Save();
            self.Test().join();
            self.state = Model_State.Finished;
        
    @Check_Prerequisite
    def Test(self):
        global_Variable_Dict = self.tf_Session.run(
            fetches = self.process_Manager.global_Tensor_Dict['Variable_Dict'],
            )   #global variable must be got before thread.

        test_Thread = Thread(target=self._Test, args=(global_Variable_Dict,));
        test_Thread.daemon = True;
        test_Thread.start();

        return test_Thread;

    @Check_Prerequisite
    def _Test(self, global_Variable_Dict):
        test_Result_DataFrame_Dict = {};    #key: test_Name
        for test_Name, test_Pattern_List in self.pattern_Manager.test_Pattern_Dict.items():
            result_List_Dict = {};  #key: tensor_Name
            for process_Name, pattern_Name, fetches, feed_Dict, attachment_Column_Dict in test_Pattern_List:
                result_Dict = self.tf_Session.run(
                    fetches = fetches,
                    feed_dict = feed_Dict
                    )

                for overlapped_key in set(result_Dict.keys()) & set(attachment_Column_Dict.keys()):
                    result_Dict["{}.Result".format(overlapped_key)] = result_Dict.pop(overlapped_key)
                    attachment_Column_Dict["{}.Pattern".format(overlapped_key)] = attachment_Column_Dict.pop(overlapped_key)
                result_Dict.update(attachment_Column_Dict);

                for column_Name, result in result_Dict.items():
                    if not column_Name in result_List_Dict.keys():
                        result_List_Dict[column_Name] = [];
                    result_List_Dict[column_Name].append(result);

            test_Result_DataFrame_Dict[test_Name] = pd.DataFrame({
                column_Name: list(np.concatenate(result_List, axis=0))
                for column_Name, result_List in result_List_Dict.items()
                })

        self.Result_Save(global_Variable_Dict, test_Result_DataFrame_Dict);

    @Check_Prerequisite
    def Result_Save(self, global_Variable_Dict, test_Result_DataFrame_Dict):
        save_Path = os.path.join(self.save_Path, "Result").replace("\\", "/");
        if not os.path.exists(save_Path):
            os.makedirs(save_Path);

        export_Dict = {};
        export_Dict.update(global_Variable_Dict);
        export_Dict.update({"Result": test_Result_DataFrame_Dict});
        
        save_File_Path = os.path.join(
            save_Path,
            "GE_{0:07d}.LI_{1:02d}.LE_{2:07d}.pickle".format(
                global_Variable_Dict["global_Epoch"],
                global_Variable_Dict["learning_Flow_Index"],
                global_Variable_Dict["local_Epoch"]
                )
            ).replace("\\", "/")

        with open(save_File_Path,"wb") as f:
            pickle.dump(export_Dict, f, protocol= 2);

    def Get_Weight(self, variable_Name):
        try:
            return True, self.tf_Session.run([v for v in tf.trainable_variables() if v.name == variable_Name][0]);
        except Exception as e:
            return False, e;
    
    @Check_Prerequisite
    def Checkpoint_Save(self):
        try:
            save_Path = os.path.join(self.save_Path, "Checkpoint").replace("\\", "/");
            if not os.path.exists(save_Path):
                os.makedirs(save_Path);

            global_Variable_Dict = self.tf_Session.run(fetches= self.process_Manager.global_Tensor_Dict['Variable_Dict']);
            save_File_Path = os.path.join(
                save_Path,
                "GE_{0:07d}.LI_{1:02d}.LE_{2:07d}".format(
                    global_Variable_Dict["global_Epoch"],
                    global_Variable_Dict["learning_Flow_Index"],
                    global_Variable_Dict["local_Epoch"]
                    )
                ).replace("\\", "/")

            self.tf_Saver.save(
                self.tf_Session,
                save_path= save_File_Path
                )
            return True, None;
        except Exception as e:
            return False, e;
    
    def Checkpoint_Load(self, load_File_Path):
        try:
            self.tf_Saver.restore(self.tf_Session, load_File_Path);
            return True, None;
        except Exception as e:
            return False, e;

    @Check_Prerequisite
    def Prerequisite_Save(self):
        save_Path = os.path.join(self.save_Path, "Model").replace("\\", "/");
        if not os.path.exists(save_Path):
            os.makedirs(save_Path);            
        save_File_Path = os.path.join(save_Path, "Prerequisite.pickle").replace("\\", "/");

        save_Dict = {
            "Process_Name_List": list(self.process_Manager.tensor_Dict.keys()),
            "Graph_History": self.process_Manager.graph_History,
            "Pattern_Dict": self.pattern_Manager.pattern_Dict,
            "Learning_List": self.learning_Manager.learning_List,
            "Test_Dict": self.learning_Manager.test_Dict,
            }

        with open(save_File_Path, "wb") as f:
            pickle.dump(save_Dict, f, protocol=2)

    def Prerequisite_Load(
        self,
        load_Prerequisite_File_Path = None,
        load_Checkpoint_File_Path = None
        ):
        try:
            if load_Prerequisite_File_Path is None:
                load_Prerequisite_File_Path = os.path.join(self.save_Path, "Model", "Prerequisite.pickle").replace("\\", "/");

            with open(load_Prerequisite_File_Path, "rb") as f:
                load_Dict = pickle.load(f);

            self.process_Manager._Process_Load(load_Dict["Process_Name_List"], load_Dict["Graph_History"]);
            self.process_Manager.Tensor_Initialize();
            self.Run_Pattern_Manager();
            self.pattern_Manager.pattern_Dict.update(load_Dict["Pattern_Dict"]);
            self.Run_Learning_Manager();
            self.learning_Manager._Learning_Load(load_Dict["Learning_List"], load_Dict["Test_Dict"]);
            self.learning_Manager.Lock();

            global_Variable_Dict = {
                "global_Step": 0,
                "global_Epoch": 0,
                "local_Step": 0,
                "local_Epoch": 0,
                "learning_Flow_Index": 0,
                }
            if not load_Checkpoint_File_Path is None:                
                checkpoint_Load_Response, e =  self.Checkpoint_Load(load_File_Path= load_Checkpoint_File_Path);
                if not checkpoint_Load_Response:
                    raise e
                global_Variable_Dict = self.tf_Session.run(fetches = self.process_Manager.global_Tensor_Dict['Variable_Dict']);

            self.pattern_Manager.Pattern_Generate(self.learning_Manager, **global_Variable_Dict);            
            return True, None;
        except Exception as e:
            return False, e;
        
    @Check_Prerequisite
    def Prerequisite_Script_Save(self, pattern_Export= True):
        save_Path = os.path.join(self.save_Path, "Model").replace("\\", "/");
        if not os.path.exists(save_Path):
            os.makedirs(save_Path);            
        save_File_Path = os.path.join(save_Path, "Prerequisite.txt").replace("\\", "/");

        from HNet2_Prerequisite_Scripter import Generate_Script_Full;
        Generate_Script_Full(
            process_Name_List= list(self.process_Manager.tensor_Dict.keys()),
            graph_History= self.process_Manager.graph_History,
            pattern_Dict= self.pattern_Manager.pattern_Dict,
            learning_List= self.learning_Manager.learning_List,
            test_Dict= self.learning_Manager.test_Dict,
            export_Path= save_File_Path,
            pattern_Export= pattern_Export
            )

    def Prerequisite_Script_Load(
        self,
        load_Prerequisite_File_Path = None,
        load_Checkpoint_File_Path = None
        ):
        try:
            if load_Prerequisite_File_Path is None:
                load_Prerequisite_File_Path = os.path.join(self.save_Path, "Model", "Prerequisite.txt").replace("\\", "/");

            with open(load_Prerequisite_File_Path, "r") as f:
                load_Script = f.readlines();
            
            exec(''.join(load_Script).replace('hNet.', 'self.'));
            
            global_Variable_Dict = {
                "global_Step": 0,
                "global_Epoch": 0,
                "local_Step": 0,
                "local_Epoch": 0,
                "learning_Flow_Index": 0,
                }
            if not load_Checkpoint_File_Path is None:                
                checkpoint_Load_Response, e =  self.Checkpoint_Load(load_File_Path= load_Checkpoint_File_Path);
                if not checkpoint_Load_Response:
                    raise e
                global_Variable_Dict = self.tf_Session.run(fetches = self.process_Manager.global_Tensor_Dict['Variable_Dict']);

            self.pattern_Manager.Pattern_Generate(self.learning_Manager, **global_Variable_Dict);
            return True, None;
        except Exception as e:
            return False, e;