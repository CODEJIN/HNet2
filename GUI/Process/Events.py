from PyQt5 import QtCore, QtGui, QtWidgets;
import sys, os, math, importlib;
import networkx as nx
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

from HNet_Enum import Tensor_Type;
from HNet2_Prerequisite_Scripter import Generate_Script_Process;
from GUI import RegEx;
from GUI.GUI_Enum import Func_Type, Parameter_Type;
from GUI.Process.Process_Func import func_Dict;
from GUI.Process import Parameter_Widgets, Shortcut_Tab;

def Global_Assign(hNet, window_Dict, ui):
    global form_Layout_Widget, network_Graph_Canvas, tensorflow_Func_Dict, get_Parameter_Func_Dict, current_Tensor_Type;

    form_Layout_Widget = ui.formLayoutWidget;

    fig = plt.figure(figsize=(10, 10));
    network_Graph_Canvas = FigureCanvas(fig);
    network_Graph_Canvas.setParent(ui.verticalLayoutWidget);
    network_Graph_Canvas.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding);
    network_Graph_Canvas.updateGeometry();
    ui.process_GraphicsLayout.addWidget(network_Graph_Canvas);

    tensorflow_Func_Dict = {};
    get_Parameter_Func_Dict = {};
    current_Tensor_Type = None;
    required_Parameter_List = None;

def Connect(hNet, window_Dict, ui):
    Global_Assign(hNet, window_Dict, ui);

    Process_Assign(hNet, ui);

    combobox_Dict = {
        Func_Type.Placeholder: ui.placeholder_ComboBox,
        Func_Type.Forward: ui.forward_ComboBox,
        Func_Type.RNN: ui.rnn_ComboBox,
        Func_Type.Activation_Calc: ui.activation_Calc_ComboBox,
        Func_Type.Reshape: ui.reshape_ComboBox,
        Func_Type.Fundamental_Math: ui.fundamental_Math_ComboBox,
        Func_Type.Loss_Calc: ui.loss_Calc_ComboBox,
        Func_Type.Optimizer: ui.optimizer_ComboBox,
        Func_Type.Test_Calc: ui.test_Calc_ComboBox,
        Func_Type.Create: ui.create_ComboBox,
        Func_Type.Other: ui.other_ComboBox,
        }

    Function_Assign(combobox_Dict);
    Parameter_Display_Assign(hNet, combobox_Dict, ui);

    ui.function_Add_PushButton.clicked.connect(lambda: Function_Add_Event(hNet, ui, combobox_Dict));
    ui.function_Add_PushButton.clicked.connect(lambda: Network_Graph_Refresh_Event(hNet, ui));
    ui.function_Add_PushButton.clicked.connect(lambda: Script_Refresh_Event(hNet, ui));
    ui.next_PushButton.clicked.connect(lambda: Next_Button_Event(hNet, window_Dict));

    #This should be at last.
    Shortcut_Assign(hNet, ui);
    
def Process_Assign(hNet, ui):
    global network_Graph_Canvas;
    
    ui.process_Add_PushButton.clicked.connect(lambda: Process_Add_Event(hNet, ui));
    ui.process_Add_PushButton.clicked.connect(lambda: Process_Refresh_Event(hNet, ui));
    ui.process_ListWidget.currentRowChanged.connect(lambda: Process_List_Row_Changed_Event(hNet, ui));
    ui.process_ListWidget.currentRowChanged.connect(lambda: Network_Graph_Refresh_Event(hNet, ui));
    ui.process_ListWidget.currentRowChanged.connect(lambda: Script_Refresh_Event(hNet, ui));
    ui.process_LineEdit.setValidator(RegEx.Letter)

    ui.script_Save_PushButton.clicked.connect(lambda: Script_Save_Event(hNet, ui));
    ui.script_Load_PushButton.clicked.connect(lambda: Script_Load_Event(hNet, ui));
    ui.script_Load_PushButton.clicked.connect(lambda: Process_Refresh_Event(hNet, ui));

def Function_Assign(combobox_Dict):
    for combobox in combobox_Dict.values():
        combobox.addItem("");

    for function_Name, function_Info in func_Dict.items():
        combobox_Dict[function_Info["Func_Type"]].addItem(function_Name.replace("_", " "),QtCore.QVariant(function_Info));
    
def Parameter_Display_Assign(hNet, combobox_Dict, ui):
    def Make_Lambda(focused_Combobox, combobox_Dict, ui):
        '''
        The direct using lambda with for~loop makes bind problem.
        See: https://stackoverflow.com/questions/19837486/python-lambda-in-a-loop
        '''
        return lambda: Function_Combobox_Index_Changed_Event(
            hNet= hNet,
            focused_Combobox= focused_Combobox,
            combobox_Dict= combobox_Dict,
            ui= ui
            )

    for combobox in combobox_Dict.values():
        combobox.currentIndexChanged.connect(Make_Lambda(
            focused_Combobox= combobox,
            combobox_Dict= combobox_Dict,
            ui= ui
            ))

def Shortcut_Assign(hNet, ui):
    shortcut_Module_Path = os.path.join(os.getcwd(), 'GUI', 'Process', 'Shortcut').replace('\\', '/');
    sys.path.append(shortcut_Module_Path)

    shortcut_Tab_List = [];
    add_PushButton_List = [];
    for module_Name in [x[:-3] for x in os.listdir(shortcut_Module_Path) if x.endswith('.py')]:        
        graphic_File = '{}/{}.{}'.format(shortcut_Module_Path, module_Name, 'png');
        if not os.path.isfile(graphic_File):
            graphic_File = None;

        new_Shortcut_Tab, new_Add_PushButton = Shortcut_Tab.Generate_Tab(hNet, ui, module_Name, graphic_File);
        shortcut_Tab_List.append(new_Shortcut_Tab);
        add_PushButton_List.append(new_Add_PushButton);

    def Set_Tab_Enabled(shortcut_Tab_List):
        if ui.process_ListWidget.currentRow() < 0:
            return;
        process_Name= ui.process_ListWidget.currentItem().text();

        for shortcut_Tab in shortcut_Tab_List:
            shortcut_Tab.setEnabled(len(hNet.process_Manager.tensor_Dict[process_Name]) == 0)

    for add_PushButton in add_PushButton_List:
        add_PushButton.clicked.connect(lambda: Network_Graph_Refresh_Event(hNet, ui));
        add_PushButton.clicked.connect(lambda: Script_Refresh_Event(hNet, ui));
        add_PushButton.clicked.connect(lambda: Set_Tab_Enabled(shortcut_Tab_List));        
    ui.process_ListWidget.currentRowChanged.connect(lambda: Set_Tab_Enabled(shortcut_Tab_List));
    ui.function_Add_PushButton.clicked.connect(lambda: Set_Tab_Enabled(shortcut_Tab_List));

    ui.tesnor_Tab_Widget.setCurrentIndex(0);

def Network_Graph_Refresh_Event(hNet, ui):
    if ui.process_ListWidget.currentRow() < 0:
        return;

    global network_Graph_Canvas;
        
    process_Name = ui.process_ListWidget.currentItem().text();
    nx_Graph, label_Dict, color_List = hNet.process_Manager.Network_Graph(process_Name);
    plt.gca().clear();
    pos = nx.spring_layout(nx_Graph, scale=2, k=3/math.sqrt(nx_Graph.order() + 1));
    options = {
        'node_color': color_List,
        'node_size': 2000,
        'font_size': 7, 
        'width': 1,
        'arrowstyle': '-|>',
        'arrowsize': 20,
        'labels': label_Dict
        }
    nx.draw(nx_Graph, pos, **options);
    plt.tight_layout();
    network_Graph_Canvas.draw();

def Script_Refresh_Event(hNet, ui):    
    ui.script_PlainTextEdit.clear();
    ui.script_PlainTextEdit.setPlainText('\n'.join(Generate_Script_Process(
        process_Name_List= list(hNet.process_Manager.tensor_Dict.keys()),
        graph_History= hNet.process_Manager.graph_History
        )[:-2]))

def Process_Add_Event(hNet, ui):
    new_Process_Name = ui.process_LineEdit.text();
    
    if new_Process_Name == "":
        ui.process_LineEdit.setFocus();
        return;

    response, e = hNet.process_Manager.Set_New_Process(new_Process_Name);
    if not response:            
        QtWidgets.QMessageBox.critical(
            None,
            'Error!',
            'An error occurred. '            
            'Please check the error message below. '
            '\n\n{}: {}'.format(type(e).__name__, e)
            )
        return;

    ui.process_LineEdit.setText('');

    ui.script_Load_PushButton.setEnabled(False);

def Process_Refresh_Event(hNet, ui):
    ui.process_ListWidget.clear();
    for process_Name in hNet.process_Manager.tensor_Dict.keys():
        ui.process_ListWidget.addItem(process_Name);

    ui.process_ListWidget.setCurrentRow(ui.process_ListWidget.count() - 1);

def Process_List_Row_Changed_Event(hNet, ui):
    ui.tesnor_Tab_Widget.setEnabled(ui.process_ListWidget.currentRow() > -1);
    ui.custom_Tab.setEnabled(ui.process_ListWidget.currentRow() > -1);

    ui.placeholder_ComboBox.setFocus();
    ui.placeholder_ComboBox.setCurrentIndex(0);

def Script_Save_Event(hNet, ui):        
    save_File_Path = os.path.join(hNet.save_Path, 'Model', 'Process_Script.txt').replace("\\", "/");
    os.makedirs(os.path.dirname(save_File_Path), exist_ok= True)
    
    script = '\n'.join(Generate_Script_Process(
        process_Name_List= list(hNet.process_Manager.tensor_Dict.keys()),
        graph_History= hNet.process_Manager.graph_History
        )[:-2])
    with open(save_File_Path, 'w') as f:
        f.write(script);

def Script_Load_Event(hNet, ui):
    if len(hNet.process_Manager.tensor_Dict) > 0:
        QtWidgets.QMessageBox.critical(
            None,
            'Error!',
            'One or more processes already exist.'
            )
        return;

    new_FileDialog = QtWidgets.QFileDialog();
    file_Path = new_FileDialog.getOpenFileName(filter= "Process script file (*.txt)")[0];
    if file_Path == '':
        return
    with open(file_Path, "r") as f:
        load_Script = f.readlines();

    import tensorflow as tf;
    import Compatible_Functions;

    exec(''.join(load_Script));

def Function_Combobox_Index_Changed_Event(hNet, focused_Combobox, combobox_Dict, ui):    
    global form_Layout_Widget, tensorflow_Func_Dict, get_Parameter_Func_Dict, current_Tensor_Type, required_Parameter_List;

    if focused_Combobox.currentIndex() == 0:
        ui.function_Description_PlainTextEdit.setPlainText("")
        ui.function_Add_PushButton.setEnabled(False)
        form_Layout_Widget.setParent(None)
        return;

    for combobox in combobox_Dict.values():
        if focused_Combobox == combobox:
            continue;
        combobox.setCurrentIndex(0);

    function_Info = focused_Combobox.itemData(focused_Combobox.currentIndex());
    current_Tensor_Type = function_Info["Tensor_Type"];
    required_Parameter_List = function_Info["Required"];
    
    ui.function_Description_PlainTextEdit.setPlainText(function_Info["Description"].strip())

    new_FormLayoutWidget, new_FormLayout = Parameter_Widgets.Form_Layout(ui.custom_Tab);
    tensorflow_Func_Dict = {};
    get_Parameter_Func_Dict = {};
    for row_Index, (parameter_Name, parameter_Type) in enumerate(function_Info["Parameter"].items()):        
        if parameter_Name in ["tensor_Func", "rnn_Cell_Func", "state_Func", "loss_Func", "optimizer_Func"]:
            tensorflow_Func_Dict[parameter_Name] = parameter_Type;
            continue;

        process_Name = ui.process_ListWidget.currentItem().text();
        
        tensor_List = [
            (process_Name, tensor_Name) for process_Name in hNet.process_Manager.tensor_Dict.keys() for tensor_Name in hNet.process_Manager.tensor_Dict[process_Name].keys()
            if not tensor_Name in hNet.process_Manager.loss_Dict[process_Name].keys() and not '.Loss' in tensor_Name and not '.Optimizer' in tensor_Name 
            ]   

        get_Parameter_Func_Dict[parameter_Name] = Parameter_Widgets.Set_Widget(
            formLayoutWidget= new_FormLayoutWidget,
            formLayout= new_FormLayout,
            location= row_Index,
            label_Text= parameter_Name,
            parameter_Type= parameter_Type,
            process_Name= process_Name,
            tensor_List= tensor_List
            )

    form_Layout_Widget.setParent(None);
    form_Layout_Widget = new_FormLayoutWidget;
    form_Layout_Widget.show()

    ui.function_Add_PushButton.setEnabled(True)

def Function_Add_Event(hNet, ui, combobox_Dict):
    global form_Layout_Widget, tensorflow_Func_Dict, get_Parameter_Func_Dict, current_Tensor_Type, required_Parameter_List;

    if current_Tensor_Type == Tensor_Type.Placeholder:
        tensor_Func = hNet.process_Manager.Placeholder_Generate;
    elif current_Tensor_Type == Tensor_Type.Tensor:
        tensor_Func = hNet.process_Manager.Tensor_Generate;
    elif current_Tensor_Type == Tensor_Type.RNN_Tensor:
        tensor_Func = hNet.process_Manager.RNN_Tensor_Generate;
    elif current_Tensor_Type == Tensor_Type.Create:
        tensor_Func = hNet.process_Manager.Create_Tensor_Generate;
    elif current_Tensor_Type == Tensor_Type.Loss:
        tensor_Func = hNet.process_Manager.Loss_Generate;
    elif current_Tensor_Type == Tensor_Type.Optimizer:
        tensor_Func = hNet.process_Manager.Optimizer_Generate;

    parameter_Dict = {'process_Name': ui.process_ListWidget.currentItem().text()};
    parameter_Dict.update(tensorflow_Func_Dict);
    parameter_Dict.update({key: value() for key, value in get_Parameter_Func_Dict.items()});

    for parameter_Name in required_Parameter_List:
        if parameter_Dict[parameter_Name] is None or \
           (type(parameter_Dict[parameter_Name]) == str and parameter_Dict[parameter_Name].strip() == '') or \
           (type(parameter_Dict[parameter_Name]) == list and len(parameter_Dict[parameter_Name]) == 0):
            QtWidgets.QMessageBox.critical(
                None,
                'Error!',
                'Parameter \'{}\' is required.'.format(parameter_Name)
                )
            return;

    response, e = tensor_Func(**parameter_Dict);
    if not response:
        QtWidgets.QMessageBox.critical(
            None,
            'Error!',
            'An error occurred. '            
            'Please check the error message below. '
            '\n\n{}: {}'.format(type(e).__name__, e)
            )
        return;

    for combobox in combobox_Dict.values():        
        combobox.setCurrentIndex(0);
    form_Layout_Widget.setParent(None);

def Next_Button_Event(hNet, window_Dict):
    if len(hNet.process_Manager.tensor_Dict) == 0:
        QtWidgets.QMessageBox.critical(
            None,
            'Error!',
            'At least one process is required.'
            )
        return;

    optimizer_Check = False;
    for tensor_Dict in hNet.process_Manager.tensor_Dict.values():
        if any(['.Optimizer' in key for key in tensor_Dict.keys()]):
            optimizer_Check = True;
            break;
    if not optimizer_Check:
        QtWidgets.QMessageBox.critical(
            None,
            'Error!',
            'At least one process with optimizer is required.'
            )
        return;

    hNet.process_Manager.Tensor_Initialize();
    hNet.Run_Pattern_Manager();
    window_Dict["Pattern"].show();
    window_Dict["Process"].close();

