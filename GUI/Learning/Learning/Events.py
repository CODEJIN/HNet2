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
    Learning_Flow_Assign(hNet, ui);
    Matching_Assign(hNet, ui);

    ui.info_TabWidget.currentChanged.connect(lambda: Info_Tab_Index_Changed_Event(hNet, ui));
    ui.next_PushButton.clicked.connect(lambda: Next_Button_Event(hNet, window_Dict, ui));


def Learning_Flow_Assign(hNet, ui):
    ui.name_LineEdit.setValidator(RegEx.Letter);
    ui.epoch_LineEdit.setValidator(RegEx.Positive_Int);
    ui.checkpoint_Save_LineEdit.setValidator(RegEx.Positive_Int);
    ui.test_Interval_LineEdit.setValidator(RegEx.Positive_Int);
    ui.mini_Batch_Size_LineEdit.setValidator(RegEx.Positive_Int);

    ui.use_Probability_ComboBox.addItem("Use", QtCore.QVariant(True))
    ui.use_Probability_ComboBox.addItem("No use", QtCore.QVariant(False))
    
    ui.learning_Flow_Add_PushButton.clicked.connect(lambda: Learning_Flow_Add_Event(hNet, ui));
    ui.learning_Flow_Up_PushButton.clicked.connect(lambda: Learning_Flow_Up_Event(hNet, ui));
    ui.learning_Flow_Down_PushButton.clicked.connect(lambda: Learning_Flow_Down_Event(hNet, ui));
    ui.learning_Flow_ListWidget.itemDoubleClicked .connect(lambda: Learning_Flow_Delete_Event(hNet, ui));
    ui.learning_Flow_ListWidget.currentRowChanged.connect(lambda: Learning_Flow_List_Row_Changed_Event(hNet, ui));

def Matching_Assign(hNet, ui):
    ui.matching_Process_ComboBox.currentIndexChanged.connect(lambda: Matching_Process_and_Pattern_Change_Event(hNet, ui));
    ui.matching_Pattern_ComboBox.currentIndexChanged.connect(lambda: Matching_Process_and_Pattern_Change_Event(hNet, ui));
    ui.matching_Assign_PushButton.clicked.connect(lambda: Matching_Assign_Event(hNet, ui));    
    ui.matching_ListWidget.itemDoubleClicked.connect(lambda: Matching_Delete_Event(hNet, ui));    
    ui.matching_Info_Add_PushButton.clicked.connect(lambda: Matching_Info_Add_Event(hNet, ui));


def Learning_Flow_Add_Event(hNet, ui):
    new_Learning_Name = ui.name_LineEdit.text();
    epoch = ui.epoch_LineEdit.text();
    checkpoint_Save_Interval = ui.checkpoint_Save_LineEdit.text();
    test_Interval = ui.test_Interval_LineEdit.text();
    mini_Batch = ui.mini_Batch_Size_LineEdit.text();
    use_Probability = ui.use_Probability_ComboBox.itemData(ui.use_Probability_ComboBox.currentIndex());

    if new_Learning_Name == "":
        ui.name_LineEdit.setFocus();
        return;
    elif epoch == "":
        ui.epoch_LineEdit.setFocus();
        return;
    elif checkpoint_Save_Interval == "":
        ui.checkpoint_Save_LineEdit.setFocus();
        return;
    elif test_Interval == "":
        ui.test_Interval_LineEdit.setFocus();
        return;
    elif mini_Batch == "":
        ui.mini_Batch_Size_LineEdit.setFocus();
        return;
    
    response, e = hNet.learning_Manager.Set_New_Learning(
        learning_Name= ui.name_LineEdit.text(),
        epoch= int(epoch),
        checkpoint_Save_Interval= int(checkpoint_Save_Interval),
        test_Interval= int(test_Interval),
        mini_Batch= int(mini_Batch),
        use_Probability= use_Probability
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

    Learning_Flow_Change_Event(hNet, ui);
    ui.learning_Flow_ListWidget.setCurrentRow(ui.learning_Flow_ListWidget.count() - 1);

    ui.name_LineEdit.setText('');
    ui.epoch_LineEdit.setText('');
    ui.checkpoint_Save_LineEdit.setText('');
    ui.test_Interval_LineEdit.setText('');
    ui.mini_Batch_Size_LineEdit.setText('');
    ui.use_Probability_ComboBox.setCurrentIndex(0);

def Learning_Flow_Up_Event(hNet, ui):
    currentRow = ui.learning_Flow_ListWidget.currentRow();
    if currentRow == 0 or currentRow == -1:
        return;
        
    hNet.learning_Manager.learning_List[currentRow - 1], hNet.learning_Manager.learning_List[currentRow] = \
        hNet.learning_Manager.learning_List[currentRow], hNet.learning_Manager.learning_List[currentRow - 1];

    Learning_Flow_Change_Event(hNet, ui);
    ui.learning_Flow_ListWidget.setCurrentRow(currentRow - 1);

def Learning_Flow_Down_Event(hNet, ui):
    currentRow = ui.learning_Flow_ListWidget.currentRow();
    if currentRow == ui.learning_Flow_ListWidget.count() - 1 or currentRow == -1:
        return;
        
    hNet.learning_Manager.learning_List[currentRow + 1], hNet.learning_Manager.learning_List[currentRow] = \
        hNet.learning_Manager.learning_List[currentRow], hNet.learning_Manager.learning_List[currentRow + 1];

    Learning_Flow_Change_Event(hNet, ui);
    ui.learning_Flow_ListWidget.setCurrentRow(currentRow + 1);

def Learning_Flow_Delete_Event(hNet, ui):
    currentRow = ui.learning_Flow_ListWidget.currentRow();
    if currentRow == -1:
        return;
    
    if QtWidgets.QMessageBox.question(
        ui.centralwidget,
        'Continue?',
        'Are you sure you want to delete this learning?',
        QtWidgets.QMessageBox.Yes,
        QtWidgets.QMessageBox.No
        ) == QtWidgets.QMessageBox.No:
        return;

    del hNet.learning_Manager.learning_List[currentRow];

    Learning_Flow_Change_Event(hNet, ui);
    
    ui.learning_Flow_ListWidget.setCurrentRow(currentRow);
    if ui.learning_Flow_ListWidget.count() == 0:
        Learning_Info_Update_Event(hNet, ui);
    elif ui.learning_Flow_ListWidget.currentRow() == -1:
        ui.learning_Flow_ListWidget.setCurrentRow(currentRow - 1);

def Learning_Flow_Change_Event(hNet, ui):
    ui.learning_Flow_ListWidget.clear();
    for learning_Info in hNet.learning_Manager.learning_List:
        ui.learning_Flow_ListWidget.addItem(learning_Info['Name']);

def Learning_Flow_List_Row_Changed_Event(hNet, ui):
    ui.matching_Process_ComboBox.clear();
    ui.matching_Pattern_ComboBox.clear();
    ui.matching_Placeholder_ComboBox.clear();
    ui.matching_Pattern_Column_ComboBox.clear();
    ui.matching_Probability_Column_ComboBox.clear();

    if ui.learning_Flow_ListWidget.currentRow() == -1:
        ui.matching_GroupBox.setEnabled(False);
        return;

    ui.matching_GroupBox.setEnabled(True);
        
    ui.matching_Process_ComboBox.addItem('');
    for process_Name in hNet.process_Manager.tensor_Dict.keys():
        if '{}.Optimizer'.format(process_Name) in hNet.process_Manager.tensor_Dict[process_Name].keys():
            ui.matching_Process_ComboBox.addItem(process_Name);
        
    ui.matching_Pattern_ComboBox.addItem('');
    for pattern_Name in hNet.pattern_Manager.pattern_Dict.keys():
        ui.matching_Pattern_ComboBox.addItem(pattern_Name);
        
    Learning_Info_Update_Event(hNet, ui);


def Matching_Process_and_Pattern_Change_Event(hNet, ui):
    ui.matching_ListWidget.clear();

    if ui.matching_Process_ComboBox.currentIndex() <= 0 or ui.matching_Pattern_ComboBox.currentIndex() <= 0:        
        ui.matching_Info_Frame.setEnabled(False);
        return;

    ui.matching_Info_Frame.setEnabled(True);
    Matching_Combobox_Update_Event(hNet, ui);
    
def Matching_Combobox_Update_Event(hNet, ui):
    ui.matching_Placeholder_ComboBox.clear();
    ui.matching_Pattern_Column_ComboBox.clear();

    process_Name = ui.matching_Process_ComboBox.currentText();
    pattern_Name = ui.matching_Pattern_ComboBox.currentText();
    
    assigned_Placeholder_List = [
        ui.matching_ListWidget.item(row).data(QtCore.Qt.UserRole)[0]
        for row in range(ui.matching_ListWidget.count())
        ]

    for placeholder_Name in hNet.process_Manager.placeholder_Dict[process_Name].keys():
        if not placeholder_Name in assigned_Placeholder_List:
            ui.matching_Placeholder_ComboBox.addItem(placeholder_Name);

    for pattern_Column in hNet.pattern_Manager.pattern_Dict[pattern_Name].columns.values.tolist():
        ui.matching_Pattern_Column_ComboBox.addItem(pattern_Column);

    use_Probability = hNet.learning_Manager.learning_List[ui.learning_Flow_ListWidget.currentRow()]['Use_Probability'];
    ui.matching_Probability_Column_ComboBox.setEnabled(use_Probability);
    if use_Probability:
        for pattern_Column in hNet.pattern_Manager.pattern_Dict[pattern_Name].columns.values.tolist():
            ui.matching_Probability_Column_ComboBox.addItem(pattern_Column);

    ui.matching_Info_Add_PushButton.setEnabled(
        len(assigned_Placeholder_List) == len(hNet.process_Manager.placeholder_Dict[process_Name].keys())
        )        

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
    
def Matching_Info_Add_Event(hNet, ui):
    learning_Flow_Index = ui.learning_Flow_ListWidget.currentRow();
    use_Probability = hNet.learning_Manager.learning_List[learning_Flow_Index]["Use_Probability"];

    response, new_Matching_Info = hNet.learning_Manager.Get_Matching_Info(
        process_Name= ui.matching_Process_ComboBox.currentText(),
        pattern_Name= ui.matching_Pattern_ComboBox.currentText(),
        matching_List= [
            ui.matching_ListWidget.item(row).data(QtCore.Qt.UserRole)
            for row in range(ui.matching_ListWidget.count())
            ],
        probability_Pattern_Column_Name= \
            ui.matching_Probability_Column_ComboBox.currentText() if use_Probability else None
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

    response, e = hNet.learning_Manager.Learning_Add_Mathing_Info(
        learning_Name= ui.learning_Flow_ListWidget.item(learning_Flow_Index).text(),
        matching_Info= new_Matching_Info
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

    Learning_Flow_List_Row_Changed_Event(hNet, ui);


def Info_Tab_Index_Changed_Event(hNet, ui):    
    if ui.info_TabWidget.currentIndex() == 1:
        Process_Tab_Event(hNet, ui);
    elif ui.info_TabWidget.currentIndex() == 2:
        Pattern_Tab_Event(hNet, ui);

def Learning_Info_Update_Event(hNet, ui):
    ui.learning_Info_TreeWidget.clear();
    for learning_Info in hNet.learning_Manager.learning_List:
        learning_Info_Item = QtWidgets.QTreeWidgetItem(ui.learning_Info_TreeWidget);
        learning_Info_Item.setText(0, learning_Info['Name']);
        ui.learning_Info_TreeWidget.addTopLevelItem(learning_Info_Item)
        
        new_Sub_Item = QtWidgets.QTreeWidgetItem(learning_Info_Item);
        new_Sub_Item.setText(0, 'Epoch: {}'.format(learning_Info['Epoch']));
        learning_Info_Item.addChild(new_Sub_Item);

        new_Sub_Item = QtWidgets.QTreeWidgetItem(learning_Info_Item);
        new_Sub_Item.setText(0, 'Checkpoint interval: {}'.format(learning_Info['Checkpoint_Save_Interval']));
        learning_Info_Item.addChild(new_Sub_Item);

        new_Sub_Item = QtWidgets.QTreeWidgetItem(learning_Info_Item);
        new_Sub_Item.setText(0, 'Test interval: {}'.format(learning_Info['Test_Interval']));
        learning_Info_Item.addChild(new_Sub_Item);

        new_Sub_Item = QtWidgets.QTreeWidgetItem(learning_Info_Item);
        new_Sub_Item.setText(0, 'Mini batch size: {}'.format(learning_Info['Mini_Batch']));
        learning_Info_Item.addChild(new_Sub_Item);

        new_Sub_Item = QtWidgets.QTreeWidgetItem(learning_Info_Item);
        new_Sub_Item.setText(0, 'Use probability: {}'.format(learning_Info['Use_Probability']));
        learning_Info_Item.addChild(new_Sub_Item);

        matching_Info_List_Item = QtWidgets.QTreeWidgetItem(learning_Info_Item);
        matching_Info_List_Item.setText(0, 'Matching Info List');
        learning_Info_Item.addChild(matching_Info_List_Item);
        
        for matching_Info_Index, matching_Info in enumerate(learning_Info['Matching_Info_List']):
            matching_Info_Item = QtWidgets.QTreeWidgetItem(matching_Info_List_Item);
            matching_Info_Item.setText(0, 'Matching {}'.format(matching_Info_Index));
            matching_Info_List_Item.addChild(matching_Info_Item);

            new_Sub_Item = QtWidgets.QTreeWidgetItem(matching_Info_Item);
            new_Sub_Item.setText(0, 'Process: {}'.format(matching_Info['Process']));
            matching_Info_Item.addChild(new_Sub_Item);

            new_Sub_Item = QtWidgets.QTreeWidgetItem(matching_Info_Item);
            new_Sub_Item.setText(0, 'Pattern: {}'.format(matching_Info['Pattern']));
            matching_Info_Item.addChild(new_Sub_Item);

            new_Sub_Item = QtWidgets.QTreeWidgetItem(matching_Info_Item);
            new_Sub_Item.setText(0, 'Probability: {}'.format(matching_Info['Probability']));
            matching_Info_Item.addChild(new_Sub_Item);

            matching_List_Item = QtWidgets.QTreeWidgetItem(matching_Info_Item);
            matching_List_Item.setText(0, 'Matching list');
            matching_Info_Item.addChild(matching_List_Item);

            for placeholder_Name, pattern_Column in matching_Info['Matching_List']:
                new_Sub_Item = QtWidgets.QTreeWidgetItem(matching_List_Item);
                new_Sub_Item.setText(0, '{} ← {}'.format(placeholder_Name, pattern_Column));
                matching_List_Item.addChild(new_Sub_Item);

        if len(learning_Info['Matching_Info_List']) == 0:
            new_Sub_Item = QtWidgets.QTreeWidgetItem(matching_Info_List_Item);
            new_Sub_Item.setText(0, 'No assigned yet');
            matching_Info_List_Item.addChild(new_Sub_Item);

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
    if len(hNet.learning_Manager.learning_List) == 0:
        ui.name_LineEdit.setFocus();
        QtWidgets.QMessageBox.critical(
            None,
            'Error!',
            'At least one learning is required.'
            )
        return;

    for learning_Flow_Index, learning_Info in enumerate(hNet.learning_Manager.learning_List):
        if len(learning_Info['Matching_Info_List']) == 0:
            ui.learning_Flow_ListWidget.setCurrentRow(learning_Flow_Index);
            ui.matching_Process_ComboBox.setFocus();
            QtWidgets.QMessageBox.critical(
                None,
                'Error!',
                'No matching information is assigned to learning \'{}\'.'.format(learning_Info['Name'])
                )
            return;

    window_Dict["Learning.Test"].show();
    window_Dict["Learning.Learning"].close();