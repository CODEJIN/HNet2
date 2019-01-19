from PyQt5 import QtCore, QtGui, QtWidgets;
import os, math;
import networkx as nx
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

from HNet_Enum import Tensor_Type;
from GUI import RegEx;
from GUI.GUI_Enum import Func_Type, Parameter_Type;
from GUI.Process.Process_Func import func_Dict;
from GUI.Process import Parameter_Widgets;
from GUI.Pattern.PandasModel import PandasModel;

def Connect(hNet, window_Dict, ui):
    Test_Assign(hNet, ui);

    ui.info_TabWidget.currentChanged.connect(lambda: Info_Tab_Index_Changed_Event(hNet, ui));
    ui.next_PushButton.clicked.connect(lambda: Next_Button_Event(hNet, window_Dict, ui));

def Test_Assign(hNet, ui):
    ui.name_LineEdit.setValidator(RegEx.Letter);
    ui.mini_Batch_Size_LineEdit.setValidator(RegEx.Positive_Int);
    
    ui.name_LineEdit.textChanged.connect(lambda: LineEdit_Event(hNet, ui));
    ui.mini_Batch_Size_LineEdit.textChanged.connect(lambda: LineEdit_Event(hNet, ui));
    ui.process_ComboBox.currentIndexChanged.connect(lambda: Process_Assign_Event(hNet, ui));
    ui.pattern_ComboBox.currentIndexChanged.connect(lambda: Pattern_Assign_Event(hNet, ui));
    ui.process_ComboBox.currentIndexChanged.connect(lambda: Process_and_Pattern_Assign_Event(hNet, ui));
    ui.pattern_ComboBox.currentIndexChanged.connect(lambda: Process_and_Pattern_Assign_Event(hNet, ui));

    ui.get_Tensor_Add_PushButton.clicked.connect(lambda: Get_Tensor_Add_Event(hNet, ui));
    ui.get_Tensor_ListWidget.itemDoubleClicked.connect(lambda: Get_Tensor_Delete_Event(hNet, ui));

    ui.attached_Pattern_Column_Add_PushButton.clicked.connect(lambda: Attached_Pattern_Column_Add_Event(hNet, ui));
    ui.attached_Pattern_Column_ListWidget.itemDoubleClicked.connect(lambda: Attached_Pattern_Column_Delete_Event(hNet, ui));

    ui.matching_Assign_PushButton.clicked.connect(lambda: Matching_Assign_Event(hNet, ui));
    ui.matching_ListWidget.itemDoubleClicked.connect(lambda: Matching_Delete_Event(hNet, ui));

    ui.test_Add_PushButton.clicked.connect(lambda: Test_Add_Event(hNet, ui));
    ui.test_ListWidget.itemDoubleClicked.connect(lambda: Test_Delete_Event(hNet, ui));

def LineEdit_Event(hNet, ui):
    if ui.name_LineEdit.text() == "" or ui.mini_Batch_Size_LineEdit.text() == "":
        ui.process_ComboBox.clear();
        ui.pattern_ComboBox.clear();
        ui.test_Frame.setEnabled(False);
        return;
    elif ui.process_ComboBox.count() > 0 and ui.pattern_ComboBox.count() > 0:
        return;

    ui.test_Frame.setEnabled(True);
    ui.process_ComboBox.addItem('');
    for process_Name in hNet.process_Manager.placeholder_Dict.keys():
        ui.process_ComboBox.addItem(process_Name);
        
    ui.pattern_ComboBox.addItem('');
    for pattern_Name in hNet.pattern_Manager.pattern_Dict.keys():
        ui.pattern_ComboBox.addItem(pattern_Name);

def Process_Assign_Event(hNet, ui):
    ui.get_Tensor_ListWidget.clear();
    ui.get_Tensor_ComboBox.clear();
    ui.get_Tensor_GroupBox.setEnabled(ui.process_ComboBox.currentIndex() > 0);
    if ui.process_ComboBox.currentIndex() > 0:
        Get_Tensor_Combobox_Update_Event(hNet, ui);

def Pattern_Assign_Event(hNet, ui): 
    ui.attached_Pattern_Column_ListWidget.clear();
    ui.attached_Pattern_Column_ComboBox.clear();
    ui.attached_Pattern_Column_GroupBox.setEnabled(ui.pattern_ComboBox.currentIndex() > 0);
    if ui.pattern_ComboBox.currentIndex() > 0:
        Attached_Pattern_Column_Combobox_Update_Event(hNet, ui);

def Process_and_Pattern_Assign_Event(hNet, ui):
    ui.matching_ListWidget.clear();
    ui.matching_Placeholder_ComboBox.clear();
    ui.matching_Pattern_Column_ComboBox.clear();
    
    is_Assigned = ui.process_ComboBox.currentIndex() > 0 and ui.pattern_ComboBox.currentIndex() > 0;
    ui.matching_GroupBox.setEnabled(is_Assigned);
    if is_Assigned:
        Matching_Combobox_Update_Event(hNet, ui);
        
def Get_Tensor_Combobox_Update_Event(hNet, ui):
    current_Index = ui.get_Tensor_ComboBox.currentIndex();

    ui.get_Tensor_ComboBox.clear();

    process_Name = ui.process_ComboBox.currentText();

    assigned_Tensor_List = [
        ui.get_Tensor_ListWidget.item(row).text()
        for row in range(ui.get_Tensor_ListWidget.count())
        ]

    tensor_List = [
        key for key in hNet.process_Manager.tensor_Dict[process_Name].keys()
        if not key in hNet.process_Manager.loss_Dict[process_Name].keys() and
           not key in assigned_Tensor_List and
           not '.Loss' in key and
           not '.Optimizer' in key
        ]
    for tensor_Name in tensor_List:
        ui.get_Tensor_ComboBox.addItem(tensor_Name);

    ui.get_Tensor_ComboBox.setCurrentIndex(current_Index);

def Attached_Pattern_Column_Combobox_Update_Event(hNet, ui):
    current_Index = ui.attached_Pattern_Column_ComboBox.currentIndex();

    ui.attached_Pattern_Column_ComboBox.clear();

    pattern_Name = ui.pattern_ComboBox.currentText();

    assigned_Pattern_Column_List = [
        ui.attached_Pattern_Column_ListWidget.item(row).text()
        for row in range(ui.attached_Pattern_Column_ListWidget.count())
        ]

    for pattern_Column in hNet.pattern_Manager.pattern_Dict[pattern_Name].columns.values.tolist():
        if not pattern_Column in assigned_Pattern_Column_List:
            ui.attached_Pattern_Column_ComboBox.addItem(pattern_Column);

    ui.attached_Pattern_Column_ComboBox.setCurrentIndex(current_Index);

def Matching_Combobox_Update_Event(hNet, ui):
    ui.matching_Placeholder_ComboBox.clear();
    ui.matching_Pattern_Column_ComboBox.clear();

    process_Name = ui.process_ComboBox.currentText();
    pattern_Name = ui.pattern_ComboBox.currentText();
    
    assigned_Placeholder_List = [
        ui.matching_ListWidget.item(row).data(QtCore.Qt.UserRole)[0]
        for row in range(ui.matching_ListWidget.count())
        ]
    
    for placeholder_Name in hNet.process_Manager.placeholder_Dict[process_Name].keys():
        if not placeholder_Name in assigned_Placeholder_List:
            ui.matching_Placeholder_ComboBox.addItem(placeholder_Name);

    for pattern_Column in hNet.pattern_Manager.pattern_Dict[pattern_Name].columns.values.tolist():
        ui.matching_Pattern_Column_ComboBox.addItem(pattern_Column);


def Get_Tensor_Add_Event(hNet, ui):
    if ui.get_Tensor_ComboBox.currentIndex() < 0:
        return;
    ui.get_Tensor_ListWidget.addItem(ui.get_Tensor_ComboBox.currentText());
    Get_Tensor_Combobox_Update_Event(hNet, ui);

def Get_Tensor_Delete_Event(hNet, ui):
    currentRow = ui.get_Tensor_ListWidget.currentRow();
    if currentRow == -1:
        return;
        
    ui.get_Tensor_ListWidget.takeItem(currentRow);

    Get_Tensor_Combobox_Update_Event(hNet, ui);
        
    ui.get_Tensor_ListWidget.setCurrentRow(currentRow);
    if ui.get_Tensor_ListWidget.currentRow() == -1:
        ui.get_Tensor_ListWidget.setCurrentRow(currentRow - 1);


def Attached_Pattern_Column_Add_Event(hNet, ui):
    if ui.attached_Pattern_Column_ComboBox.currentIndex() < 0:
        return;
    ui.attached_Pattern_Column_ListWidget.addItem(ui.attached_Pattern_Column_ComboBox.currentText());
    Attached_Pattern_Column_Combobox_Update_Event(hNet, ui)

def Attached_Pattern_Column_Delete_Event(hNet, ui):
    currentRow = ui.attached_Pattern_Column_ListWidget.currentRow();
    if currentRow == -1:
        return;
        
    ui.attached_Pattern_Column_ListWidget.takeItem(currentRow);

    Attached_Pattern_Column_Combobox_Update_Event(hNet, ui);
        
    ui.attached_Pattern_Column_ListWidget.setCurrentRow(currentRow);
    if ui.attached_Pattern_Column_ListWidget.currentRow() == -1:
        ui.attached_Pattern_Column_ListWidget.setCurrentRow(currentRow - 1);


def Matching_Assign_Event(hNet, ui):
    if ui.matching_Placeholder_ComboBox.currentIndex() < 0 or ui.matching_Pattern_Column_ComboBox.currentIndex() < 0:
        return;

    placeholder_Name = ui.matching_Placeholder_ComboBox.currentText();
    pattern_Column = ui.matching_Pattern_Column_ComboBox.currentText();
    
    new_Item = QtWidgets.QListWidgetItem('{} ← {}'.format(placeholder_Name, pattern_Column), ui.matching_ListWidget);
    new_Item.setData(QtCore.Qt.UserRole, (placeholder_Name, pattern_Column));
    ui.matching_ListWidget.addItem(new_Item);

    Matching_Combobox_Update_Event(hNet, ui);
       
def Matching_Delete_Event(hNet, ui):
    currentRow = ui.matching_ListWidget.currentRow();
    if currentRow == -1:
        return;

    ui.matching_ListWidget.takeItem(currentRow);

    Matching_Combobox_Update_Event(hNet, ui);

    ui.matching_ListWidget.setCurrentRow(currentRow);
    if ui.matching_ListWidget.count() > 0 and ui.matching_ListWidget.currentRow() == -1:
        ui.matching_ListWidget.setCurrentRow(currentRow - 1);


def Test_Add_Event(hNet, ui):
    if ui.process_ComboBox.currentText() == '':
        ui.process_ComboBox.setFocus();
        return;
    elif ui.pattern_ComboBox.currentText() == '':
        ui.pattern_ComboBox.setFocus();
        return;
    elif ui.get_Tensor_ListWidget.count() == 0:
        ui.get_Tensor_ComboBox.setFocus();
        return;

    process_Name = ui.process_ComboBox.currentText();
    pattern_Name = ui.pattern_ComboBox.currentText();

    get_Tensor_Name_List = [
        ui.get_Tensor_ListWidget.item(row).text()
        for row in range(ui.get_Tensor_ListWidget.count())
        ]

    matching_List = [
        ui.matching_ListWidget.item(row).data(QtCore.Qt.UserRole)
        for row in range(ui.matching_ListWidget.count())
        ]

    dependency_Placeholder_Name_List = [];
    for tensor_Name in get_Tensor_Name_List:
        dependency_Placeholder_Name_List.extend(hNet.process_Manager.Get_Tensor_Dependency(process_Name, tensor_Name));
    dependency_Placeholder_Name_List = list(set(dependency_Placeholder_Name_List));   

    included_Matching_List = [
        matching for matching in matching_List
        if matching[0] in dependency_Placeholder_Name_List
        ]
    excluded_Matching_List = [
        '{} ← {}'.format(matching[0], matching[1]) for matching in matching_List
        if not matching[0] in dependency_Placeholder_Name_List
        ]
    if len(excluded_Matching_List) > 0:
        QtWidgets.QMessageBox.information(
            None,
            'Warning!',
            'The following matchings were excluded because they did not affect the tensor extracting you select:\n'
            '{}'.format(excluded_Matching_List)
            )

    response, new_Matching_Info = hNet.learning_Manager.Get_Matching_Info(
        process_Name= process_Name,
        pattern_Name= pattern_Name,
        matching_List= included_Matching_List,        
        get_Tensor_Name_List = get_Tensor_Name_List
        )
    if not response:
        e = new_Matching_Info;
        QtWidgets.QMessageBox.critical(
            None,
            'Error!',
            'An error occurred. '            
            'Please check the error message below. '
            '\n\n{}: {}'.format(type(e).__name__, e)
            )
        return;

    response, e = hNet.learning_Manager.Set_New_Test(
        test_Name= ui.name_LineEdit.text(),
        mini_Batch= int(ui.mini_Batch_Size_LineEdit.text()),
        matching_Info = new_Matching_Info,
        get_Tensor_Name_List = get_Tensor_Name_List,
        attached_Pattern_Column_Name_List = [
            ui.attached_Pattern_Column_ListWidget.item(row).text()
            for row in range(ui.attached_Pattern_Column_ListWidget.count())
            ]
        )
    if not response:
        QtWidgets.QMessageBox.critical(
            None,
            'Error!',
            'An error occurred. '            
            'Please check the error message below. '
            '\n\n{}: {}'.format(type(e).__name__, e)
            )
        return;

    Test_Change_Event(hNet, ui);

def Test_Delete_Event(hNet, ui):
    currentRow = ui.test_ListWidget.currentRow();
    if currentRow == -1:
        return;
    
    if QtWidgets.QMessageBox.question(
        ui.centralwidget,
        'Continue?',
        'Are you sure you want to delete this test?',
        QtWidgets.QMessageBox.Yes,
        QtWidgets.QMessageBox.No
        ) == QtWidgets.QMessageBox.No:
        return;

    del hNet.learning_Manager.test_Dict[ui.test_ListWidget.item(currentRow).text()];

    Test_Change_Event(hNet, ui);
        
    ui.test_ListWidget.setCurrentRow(currentRow);
    if ui.test_ListWidget.currentRow() == -1:
        ui.test_ListWidget.setCurrentRow(currentRow - 1);
    Test_Info_Update_Event(hNet, ui);

def Test_Change_Event(hNet, ui):
    ui.test_ListWidget.clear();
    ui.name_LineEdit.setText("");
    ui.mini_Batch_Size_LineEdit.setText("");

    for test_Name in hNet.learning_Manager.test_Dict.keys():
        ui.test_ListWidget.addItem(test_Name);

    Test_Info_Update_Event(hNet, ui);

def Info_Tab_Index_Changed_Event(hNet, ui):    
    if ui.info_TabWidget.currentIndex() == 1:
        Process_Tab_Event(hNet, ui);
    elif ui.info_TabWidget.currentIndex() == 2:
        Pattern_Tab_Event(hNet, ui);

def Test_Info_Update_Event(hNet, ui):
    ui.test_Info_TreeWidget.clear();
    for test_Name, test_Info in hNet.learning_Manager.test_Dict.items():
        test_Info_Item = QtWidgets.QTreeWidgetItem(ui.test_Info_TreeWidget);
        test_Info_Item.setText(0, test_Name);
        ui.test_Info_TreeWidget.addTopLevelItem(test_Info_Item)
        
        new_Sub_Item = QtWidgets.QTreeWidgetItem(test_Info_Item);
        new_Sub_Item.setText(0, 'Mini batch size: {}'.format(test_Info['Mini_Batch']));
        test_Info_Item.addChild(new_Sub_Item);
        

        matching_Info = test_Info['Matching_Info'];

        matching_Info_Item = QtWidgets.QTreeWidgetItem(test_Info_Item);
        matching_Info_Item.setText(0, 'Matching info');
        test_Info_Item.addChild(matching_Info_Item);

        new_Sub_Item = QtWidgets.QTreeWidgetItem(matching_Info_Item);
        new_Sub_Item.setText(0, 'Process: {}'.format(matching_Info['Process']));
        matching_Info_Item.addChild(new_Sub_Item);

        new_Sub_Item = QtWidgets.QTreeWidgetItem(matching_Info_Item);
        new_Sub_Item.setText(0, 'Pattern: {}'.format(matching_Info['Pattern']));
        matching_Info_Item.addChild(new_Sub_Item);

        matching_List_Item = QtWidgets.QTreeWidgetItem(matching_Info_Item);
        matching_List_Item.setText(0, 'Matching list');
        matching_Info_Item.addChild(matching_List_Item);

        for placeholder_Name, pattern_Column in matching_Info['Matching_List']:
            new_Sub_Item = QtWidgets.QTreeWidgetItem(matching_List_Item);
            new_Sub_Item.setText(0, '{} ← {}'.format(placeholder_Name, pattern_Column));
            matching_List_Item.addChild(new_Sub_Item);


        get_Tensor_Name_List_Item = QtWidgets.QTreeWidgetItem(test_Info_Item);
        get_Tensor_Name_List_Item.setText(0, 'Get tensor list');
        test_Info_Item.addChild(get_Tensor_Name_List_Item);

        for tensor_Name in test_Info['Get_Tensor_Name_List']:
            new_Sub_Item = QtWidgets.QTreeWidgetItem(get_Tensor_Name_List_Item);
            new_Sub_Item.setText(0, tensor_Name);
            get_Tensor_Name_List_Item.addChild(new_Sub_Item);


        if len(test_Info['Attached_Pattern_Column_Name_List']) > 0:
            attached_Pattern_Column_List_Item = QtWidgets.QTreeWidgetItem(test_Info_Item);
            attached_Pattern_Column_List_Item.setText(0, 'Attached pattern column list');
            test_Info_Item.addChild(attached_Pattern_Column_List_Item);

            for pattern_Column in test_Info['Attached_Pattern_Column_Name_List']:
                new_Sub_Item = QtWidgets.QTreeWidgetItem(attached_Pattern_Column_List_Item);
                new_Sub_Item.setText(0, pattern_Column);
                attached_Pattern_Column_List_Item.addChild(new_Sub_Item);


def Process_Tab_Event(hNet, ui):
    #There is no showevent which can be connected. Thus, I use this check method.
    if len(hNet.process_Manager.tensor_Dict) == ui.process_TabWidget.count():
        return;
    
    #Bug? First tab page doesn't see anything. Make dummy and delete.
    ui.process_TabWidget.addTab(QtWidgets.QWidget(), "Dummy");

    for process_Name in hNet.process_Manager.tensor_Dict.keys():
        new_Process_Tab  = QtWidgets.QWidget();
        ui.process_TabWidget.addTab(new_Process_Tab, process_Name);

        new_Layout_Widget = QtWidgets.QWidget(new_Process_Tab);
        new_Layout_Widget.setGeometry(QtCore.QRect(10, 10, 531, 511));
        new_Layout = QtWidgets.QVBoxLayout(new_Layout_Widget);
        new_Layout.setContentsMargins(0, 0, 0, 0)
        
        fig = plt.figure(figsize=(10, 10));
        new_Network_Graph_Canvas = FigureCanvas(fig);
        new_Network_Graph_Canvas.setParent(new_Layout_Widget);
        new_Network_Graph_Canvas.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding);
        new_Network_Graph_Canvas.updateGeometry();

        nx_Graph, label_Dict, color_List = hNet.process_Manager.Network_Graph(process_Name);
        plt.gca().clear();
        pos = nx.spring_layout(nx_Graph, scale=2, k=3/math.sqrt(nx_Graph.order()));
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
        new_Network_Graph_Canvas.draw();

        new_Layout.addWidget(new_Network_Graph_Canvas);

    #Bug? First tab page doesn't see anything.
    ui.process_TabWidget.removeTab(0);

def Pattern_Tab_Event(hNet, ui):
    #There is no showevent which can be connected. Thus, I use this check method.
    if len(hNet.pattern_Manager.pattern_Dict) == ui.pattern_TabWidget.count():
        return;

    for pattern_Name in hNet.pattern_Manager.pattern_Dict.keys():
        new_PandasModel = PandasModel(hNet.pattern_Manager.pattern_Dict[pattern_Name].loc[:100]);

        new_Tab = QtWidgets.QWidget();
        new_TableView = QtWidgets.QTableView(new_Tab);
        new_TableView.setGeometry(QtCore.QRect(10, 10, 531, 511))    
        new_TableView.setModel(new_PandasModel)
        ui.pattern_TabWidget.addTab(new_Tab, pattern_Name);


def Next_Button_Event(hNet, window_Dict, ui):
    if len(hNet.learning_Manager.test_Dict) == 0:
        ui.name_LineEdit.setFocus();
        QtWidgets.QMessageBox.critical(
            None,
            'Error!',
            'At least one test is required.'
            )
        return;

    hNet.learning_Manager.Lock();
    hNet.pattern_Manager.Pattern_Generate(hNet.learning_Manager)
    hNet.Prerequisite_Save()
    hNet.Prerequisite_Script_Save(pattern_Export= False)


    window_Dict["Train"].show();
    window_Dict["Learning.Test"].close();

    
    
    