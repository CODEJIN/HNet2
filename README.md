# HNet2

HNet 2.0 was built and tested in Python version 3.6.4 and has not been tested to work with Python 2.x. In addition, HNet 2.0 supports GPU acceleration based on Tensorflow, so if you need GPU acceleration, please configure the environment by installing CUDA etc. beforehand.
Before starting the program, enter the following command in the terminal (command line) to complete the upgrade of modules.

    pip install -r requirements.txt
    
After completing the configuration for running HNet 2.0, execute the following command to tenimal to run HNet 2.0.

    Python hnet2_gui.py
    
HNet 2.0 is a 5-step setup including the main, and the model is learned and tested.

# Main: Workspace and Load
![main](https://user-images.githubusercontent.com/17133841/51434372-0d138580-1c2d-11e9-9e3d-938721d942aa.png)

Workspace is the folder (directory) where all the model's information is stored. In HNet 2.0, model structure information, learned weight information, and test information are recorded in workspace.
Prerequisite load is the function to load a previously configured model. The prerequisite file is a file of the model's set up information. You can load a file with '.pickle' and '.txt', which will be created in the workspace when you complete all five settings. If you import the model using the prerequisite file, skip all remaining steps and go straight to the train process. On the other hand, the checkpoint file is a file about the weight of the model that we learned before. In other words, the weights of model which is loaded by prerequisite file is changed to the state previously learned. If you only use the prerequisite file, you will start learning from scratch.

# Process setup
![process](https://user-images.githubusercontent.com/17133841/51434374-0d138580-1c2d-11e9-99fd-1179818e36ca.png)

Process represents one activation flow in HNet 2.0. The user can use the shortcut tabs at the top right to load a process of a predetermined structure or freely configure a process using a custom tab. The basic elements of process are as follows. 1) A placeholder to insert input and target values (red of the graph). 2) A tensor representing the computation (blue green of the graph). 3) A loss that represents the degree how much model's result is wrong through a comparison the output tensor and the target placeholder (green of the graph). 4) An optimizer that modifies each weight to reduce future loss (purple of the graph).
Custom tap allows you to use functions that are mainly used in TensorFlow or deep learning for process configuration. All function is divided into 11 types. If you select function, a brief description of the function is displayed in the upper right corner. The user can determine the detailed parameters of each function and add them to the process.

# Pattern setup
![pattern](https://user-images.githubusercontent.com/17133841/51434373-0d138580-1c2d-11e9-8b42-112eb32eb79c.png)

HNet 2.0 receives pickled Pandas dataframe as a pattern. If you set the pattern file and the name in the model of the pattern, you can see brief information of the pattern inputted at the bottom.

## Pattern example (MNIST)

https://drive.google.com/drive/folders/1zYmzUpIL_EzgNcbpcRMOkPHyQ0p0MnC8

# Learning setup
![learning](https://user-images.githubusercontent.com/17133841/51434371-0d138580-1c2d-11e9-8494-fe76eccd1d13.png)

The process is the specific way of learning, and the pattern is what you want model to learn. And a learning matches them and sets how much model learn them. In learning, you will determine how many epochs will be trained during learning, and how often you want to store checkpoint and test in the middle of learning. And, learning will match which pattern column entered in the process will be entered in the placeholder.
The entered learning information can be checked on the right panel, and the process and pattern can be checked again using the same tab.

# Test setup
![test](https://user-images.githubusercontent.com/17133841/51434375-0d138580-1c2d-11e9-82ba-66787632860a.png)

Test determines what information is extracted from the model during learning and at the end. If necessary, you can use the same or different process as learning, and you can extract most tensors in the process.
The inserted test information can be checked in the right panel like learning, and process and pattern can be checked again by using tabs in the same position.

# Train
![train1](https://user-images.githubusercontent.com/17133841/51434370-0c7aef00-1c2d-11e9-9165-d4f6cb8896ab.png)
![train2](https://user-images.githubusercontent.com/17133841/51434376-0d138580-1c2d-11e9-8d9c-f8be0b6bf14e.png)

When the whole setting is finished, the train screen will be presented, and the learning will proceed. During learning, HNet 2.0 graphs current learning progress and simple loss variation. Thus, you can keep track of model's training situation. In addition, you can use the tabs at the top to see the model's current weights structure.
