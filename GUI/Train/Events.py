from PyQt5 import QtCore, QtGui, QtWidgets;
import os, time, math;
import networkx as nx
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import numpy as np;
import tensorflow as tf;
from HNet_Enum import Model_State;
from GUI.Pattern.PandasModel import PandasModel;

def Connect(hNet, window_Dict, ui):
    ui.info_TabWidget.currentChanged.connect(lambda: Info_Tab_Index_Changed_Event(hNet, ui));
    
    ui.weight_ComboBox.currentIndexChanged.connect(lambda: Weight_Combobox_Event(hNet, ui));
    ui.filter_Indexing_From_VerticalSlider.valueChanged.connect(lambda: From_VerticalSlider_Event(hNet, ui));
    ui.filter_Indexing_To_VerticalSlider.valueChanged.connect(lambda: To_VerticalSlider_Event(hNet, ui));
    
    ui.start_PushButton.clicked.connect(lambda: Start_Button_Event(hNet, ui));
    ui.checkpoint_Save_PushButton.clicked.connect(lambda: Checkpoint_Save_Button_Event(hNet));


def Update(hNet, ui):
    for index in range(ui.loss_Flow_Layout.count()):
        ui.loss_Flow_Layout.itemAt(index).widget().setParent(None);
    for index in range(ui.weight_Layout.count()):
        ui.weight_Layout.itemAt(index).widget().setParent(None);

    loss_Flow_Graph_Figure = plt.figure(figsize=(10, 10));
    loss_Flow_Graph_Canvas = FigureCanvas(loss_Flow_Graph_Figure);
    loss_Flow_Graph_Canvas.setParent(ui.verticalLayoutWidget);
    loss_Flow_Graph_Canvas.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding);
    loss_Flow_Graph_Canvas.updateGeometry();
    ui.loss_Flow_Layout.addWidget(loss_Flow_Graph_Canvas);
    loss_Flow_Graph_X_Lim = sum([learning_Info['Epoch'] for learning_Info in hNet.learning_Manager.learning_List])

    Weight_Tab_Event(hNet, ui);
    weight_Image_Figure = plt.figure(figsize=(10, 10));
    weight_Image_Canvas = FigureCanvas(weight_Image_Figure);
    weight_Image_Canvas.setParent(ui.verticalLayoutWidget_2);
    weight_Image_Canvas.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding);
    weight_Image_Canvas.updateGeometry();
    ui.weight_Layout.addWidget(weight_Image_Canvas);

    #This need figure and canvas
    ui.weight_ComboBox.currentIndexChanged.connect(lambda: Update_Weight(hNet, ui, weight_Image_Figure, weight_Image_Canvas));
    ui.filter_Indexing_From_VerticalSlider.valueChanged.connect(lambda: Update_Weight(hNet, ui, weight_Image_Figure, weight_Image_Canvas));
    ui.filter_Indexing_To_VerticalSlider.valueChanged.connect(lambda: Update_Weight(hNet, ui, weight_Image_Figure, weight_Image_Canvas));

    refresh_Count = np.uint64(0);
    while not hNet.state in [Model_State.Finished, Model_State.Paused]:
        if len(hNet.training_Loss_DataFrame) == 0:
            continue;

        #QThread freeze all pyqt GUI. Thus, 'ProcessEvents' should be called frequently than real update.
        time.sleep(0.001);
        QtWidgets.QApplication.processEvents();
        
        try:    #To ignore indexing problem of thread conflict
            if refresh_Count % 100 == 0:
                Update_Train_Info(hNet, ui, loss_Flow_Graph_Figure, loss_Flow_Graph_Canvas, loss_Flow_Graph_X_Lim);
                Update_Start_Button_State(hNet, ui)
            if refresh_Count % 1000 == 0:
                Update_Weight(hNet, ui, weight_Image_Figure, weight_Image_Canvas);
        except:
            pass;

        refresh_Count += 1;

    Update_Start_Button_State(hNet, ui)
    if hNet.state == Model_State.Finished:
        Update_Train_Info(hNet, ui, loss_Flow_Graph_Figure, loss_Flow_Graph_Canvas, loss_Flow_Graph_X_Lim);
        Update_Weight(hNet, ui, weight_Image_Figure, weight_Image_Canvas);
    
        ui.train_ProgressBar.setValue(ui.train_ProgressBar.value() + 1);

        QtWidgets.QMessageBox.information(
            None,
            'Notice',
            'Training is done!'
            )
    
def Update_Train_Info(hNet, ui, loss_Flow_Graph_Figure, loss_Flow_Graph_Canvas, loss_Flow_Graph_X_Lim):
    global_Step, global_Epoch, learning_Flow_Index, local_Step, local_Epoch, process_Name, pattern_Name, loss = \
        hNet.training_Loss_DataFrame.loc[len(hNet.training_Loss_DataFrame) - 1].values;

    ui.global_Step_LineEdit.setText(str(global_Step));
    ui.global_Epoch_LineEdit.setText(str(global_Epoch));
    ui.learning_Flow_Index_LineEdit.setText(str(learning_Flow_Index));
    ui.local_Step_LineEdit.setText(str(local_Step));
    ui.local_Epoch_LineEdit.setText(str(local_Epoch));
    ui.process_Name_LineEdit.setText(process_Name);
    ui.pattern_Name_LineEdit.setText(pattern_Name);
    ui.loss_LineEdit.setText(str(np.round(loss, 5)));

    ui.train_ProgressBar.setValue(global_Epoch);

    #Flow draw;
    mean_Loss_DataFrame = hNet.training_Loss_DataFrame[['Global_Epoch', 'Loss']].groupby('Global_Epoch').mean().reset_index();    
        
    loss_Flow_Graph_Figure.clear();
    axes = loss_Flow_Graph_Figure.add_subplot(111);
    axes.plot(
        mean_Loss_DataFrame['Global_Epoch'].values,
        np.round(mean_Loss_DataFrame['Loss'].values, 3)
        )
    axes.set_xlim(0, loss_Flow_Graph_X_Lim);
    axes.set_ylim(0, np.max(mean_Loss_DataFrame['Loss'].values));    
    plt.tight_layout();
    loss_Flow_Graph_Canvas.draw();
    
def Update_Weight(hNet, ui, weight_Image_Figure, weight_Image_Canvas):
    weight = hNet.Get_Weight(ui.weight_ComboBox.currentText())[1];
    if len(weight.shape) > 2:   #Conv
        weight = weight[
            ...,
            ui.filter_Indexing_From_VerticalSlider.value(),
            ui.filter_Indexing_To_VerticalSlider.value()
            ]
    if len(weight.shape) == 1: #Bias or Conv1D
        weight = np.expand_dims(weight, axis=0);
        
    weight = np.transpose(weight);
    
    weight_Image_Figure.clear();
    axes = weight_Image_Figure.add_subplot(111);
    axes.imshow(weight, aspect='auto');
    axes.set_xticks([]);
    axes.set_yticks([]);
    axes.tick_params(axis='both', which='both', length=0)
    weight_Image_Figure.tight_layout();
    weight_Image_Canvas.draw();

def Update_Start_Button_State(hNet, ui):
    if hNet.state == Model_State.Finished:
        ui.start_PushButton.setText('Done');
        ui.start_PushButton.setEnabled(False);
        ui.checkpoint_Save_PushButton.setEnabled(True);
    elif hNet.state == Model_State.Running:
        ui.start_PushButton.setText('Pause');
        ui.start_PushButton.setEnabled(True);
        ui.checkpoint_Save_PushButton.setEnabled(False);
    elif hNet.state == Model_State.Pausing:
        ui.start_PushButton.setText('Pausing');
        ui.start_PushButton.setEnabled(False);
        ui.checkpoint_Save_PushButton.setEnabled(False);
    elif hNet.state == Model_State.Paused:
        ui.start_PushButton.setText('Resume');
        ui.start_PushButton.setEnabled(True);
        ui.checkpoint_Save_PushButton.setEnabled(True);

def Info_Tab_Index_Changed_Event(hNet, ui):    
    if ui.info_TabWidget.currentIndex() == 1:
        Weight_Tab_Event(hNet, ui);
    elif ui.info_TabWidget.currentIndex() == 2:
        Process_Tab_Event(hNet, ui);
    elif ui.info_TabWidget.currentIndex() == 3:
        Pattern_Tab_Event(hNet, ui);
    elif ui.info_TabWidget.currentIndex() == 4:
        Learning_Tab_Event(hNet, ui);
    elif ui.info_TabWidget.currentIndex() == 5:
        Test_Tab_Event(hNet, ui);

def Weight_Tab_Event(hNet, ui):
    #There is no showevent which can be connected. Thus, I use this check method.
    if len(tf.trainable_variables()) == ui.weight_ComboBox.count():
        return;

    for v in tf.trainable_variables():
        variable_Name = v.name;
        ui.weight_ComboBox.addItem(variable_Name);
    ui.weight_ComboBox.setCurrentIndex(0);

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
        new_Layout_Widget.setGeometry(QtCore.QRect(10, 10, 581, 421));
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
        new_TableView.setGeometry(QtCore.QRect(10, 10, 581, 421))    
        new_TableView.setModel(new_PandasModel)
        ui.pattern_TabWidget.addTab(new_Tab, pattern_Name);

def Learning_Tab_Event(hNet, ui):
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

def Test_Tab_Event(hNet, ui):
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

def Weight_Combobox_Event(hNet, ui):
    variable_Shape = [v for v in tf.trainable_variables() if v.name == ui.weight_ComboBox.currentText()][0].shape.as_list();
    ui.filter_Indexing_GroupBox.setEnabled(len(variable_Shape) >  2);
    if len(variable_Shape) >  2:
        ui.filter_Indexing_From_VerticalSlider.setMaximum(variable_Shape[-2] - 1)
        ui.filter_Indexing_To_VerticalSlider.setMaximum(variable_Shape[-1] - 1)

    ui.filter_Indexing_From_VerticalSlider.setValue(0);
    ui.filter_Indexing_To_VerticalSlider.setValue(0);
    
def From_VerticalSlider_Event(hNet, ui):
    ui.filter_Indexing_From_LineEdit.setText(str(ui.filter_Indexing_From_VerticalSlider.value()));

def To_VerticalSlider_Event(hNet, ui):
    ui.filter_Indexing_To_LineEdit.setText(str(ui.filter_Indexing_To_VerticalSlider.value()));

def Start_Button_Event(hNet, ui):
    if hNet.state == Model_State.Finished:
        return;
    elif hNet.state == Model_State.Paused:    #paused -> running
        hNet.Train();

        update_Thread = QtCore.QThread(ui.info_TabWidget);
        update_Thread.started.connect(lambda: Update(hNet, ui));
        update_Thread.start();
                
        Update_Start_Button_State(hNet, ui);
        
        ui.train_ProgressBar.setMaximum(
            sum([learning_Info['Epoch'] for learning_Info in hNet.learning_Manager.learning_List])
            )
    elif hNet.state == Model_State.Running:   #running -> pausing
        hNet.state = Model_State.Pausing;
        Update_Start_Button_State(hNet, ui);

def Checkpoint_Save_Button_Event(hNet):
    if not hNet.state == Model_State.Paused:
        QtWidgets.QMessageBox.critical(
            None,
            'Error!',
            'Checkpoint can be saved when only training is paused.'
            )
        return;

    response, e = hNet.Checkpoint_Save();
    if response:
        QtWidgets.QMessageBox.information(
            None,
            'Notice',
            'Checkpoint is saved.'
            )
        return;
    else:
        QtWidgets.QMessageBox.critical(
            None,
            'Error!',
            'An error occurred. '            
            'Please check the error message below. '
            '\n\n{}: {}'.format(type(e).__name__, e)
            )
        return;