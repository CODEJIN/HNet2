from PyQt5 import QtCore, QtGui, QtWidgets;
import os;

from GUI.Pattern.PandasModel import PandasModel;
from GUI import RegEx;

def Connect(hNet, window_Dict, ui):
    ui.name_LineEdit.setValidator(RegEx.Letter);

    ui.file_Browser_PushButton.clicked.connect(lambda: File_Browser_Event("Pattern pickle file (*.pickle)", ui))
    ui.add_PushButton.clicked.connect(lambda: Add_Button_Event(hNet, ui))
    ui.next_PushButton.clicked.connect(lambda: Next_Button_Event(hNet, window_Dict))
    
def File_Browser_Event(filter, ui):
    new_FileDialog = QtWidgets.QFileDialog();
    file_Path = new_FileDialog.getOpenFileName(filter= filter)[0];
    if file_Path != "":
        ui.file_LineEdit.setText(file_Path);

def Add_Button_Event(hNet, ui):
    pattern_File_Path = ui.file_LineEdit.text();
    pattern_Name = ui.name_LineEdit.text();
    if  pattern_File_Path == "":
        QtWidgets.QMessageBox.critical(
            None,
            'Error!',
            'Assign the pattern file path.'
            )
        return;
        ui.file_Browser_PushButton.setFocus();
        return;
    if pattern_Name == "":
        QtWidgets.QMessageBox.critical(
            None,
            'Error!',
            'Insert the pattern name.'
            )
        ui.name_LineEdit.setFocus();
        return;
    
    response, e = hNet.pattern_Manager.Pattern_Load(pattern_Name, pattern_File_Path);
    if not response:
        QtWidgets.QMessageBox.critical(
            None,
            'Error!',
            'An error occurred. '            
            'Please check the error message below. '
            '\n\n{}: {}'.format(type(e).__name__, e)
            )
        return;

    ui.file_LineEdit.setText('');
    ui.name_LineEdit.setText('');

    new_PandasModel = PandasModel(hNet.pattern_Manager.pattern_Dict[pattern_Name].loc[:100]);

    new_Tab = QtWidgets.QWidget();
    new_TableView = QtWidgets.QTableView(new_Tab);
    new_TableView.setGeometry(QtCore.QRect(10, 10, 681, 521))    
    new_TableView.setModel(new_PandasModel)
    ui.preview_Tab_Widget.addTab(new_Tab, pattern_Name);
    ui.preview_Tab_Widget.setCurrentIndex(ui.preview_Tab_Widget.count() - 1);

def Next_Button_Event(hNet, window_Dict):
    if len(hNet.pattern_Manager.pattern_Dict) == 0:
        ui.file_Browser_PushButton.setFocus();
        QtWidgets.QMessageBox.critical(
            None,
            'Error!',
            'At least one pattern is required.'
            )
        return;

    hNet.Run_Learning_Manager();
    window_Dict["Learning.Learning"].show();
    window_Dict["Pattern"].close();

    
    
    