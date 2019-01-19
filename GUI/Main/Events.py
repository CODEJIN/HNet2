from PyQt5 import QtCore, QtGui, QtWidgets;
import os, datetime;
from tensorflow.python.client import device_lib;

def Connect(hNet, window_Dict, ui):
    Title_Text_Assign(ui);
    Device_Check(ui);

    ui.workspace_Browser_PushButton.clicked.connect(lambda: Directory_Browser_Event(ui))
    ui.prerequisite_File_Browser_PushButton.clicked.connect(lambda: File_Browser_Event("Prerequisite file (*.pickle *.txt)", ui));
    ui.checkpoint_File_Browser_PushButton.clicked.connect(lambda: Checkpoint_Browser_Event(ui));
    ui.next_PushButton.clicked.connect(lambda: Next_Button_Event(hNet, window_Dict, ui));
    
def Title_Text_Assign(ui):
    current_Year = datetime.datetime.now().year;
    year_String = "2019{}".format("" if current_Year <= 2019 else " - {}".format(current_Year));

    ui.title_TextEdit.setHtml(
        "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
        "<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
        "p, li {{ white-space: pre-wrap; }}\n"
        "</style></head><body style=\" font-family:\'Gulim\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
        "<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:24pt;\">HNet 2.0 Alpha ver.</span></p>\n"
        "<p align=\"center\" style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p>\n"
        "<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">하아....</p>\n"
        "<p align=\"center\" style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p>\n"
        "<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Copyright {}. Heejo You. All rights reserved.</p>\n"
        "<p align=\"center\" style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>".format(year_String)
        )
   
def Device_Check(ui):
    if len([x for x in device_lib.list_local_devices() if x.device_type == 'GPU']) < 1:
        ui.device_Signal_Label.setStyleSheet('color: red')
        ui.device_Info_Label.setText('HNet could not find GPU device. HNet is working on CPU.')
    else:
        ui.device_Signal_Label.setStyleSheet('color: green')
        ui.device_Info_Label.setText('GPU accelation ready.')

def Directory_Browser_Event(ui):
    new_FileDialog = QtWidgets.QFileDialog();
    file_Path = new_FileDialog.getExistingDirectory()
    if file_Path != "":
        ui.workspace_LineEdit.setText(file_Path);

def File_Browser_Event(filter, ui):
    new_FileDialog = QtWidgets.QFileDialog();
    file_Path = new_FileDialog.getOpenFileName(filter= filter)[0];
    if file_Path != "":
        ui.prerequisite_File_LineEdit.setText(file_Path);

def Checkpoint_Browser_Event(ui):
    new_FileDialog = QtWidgets.QFileDialog();
    file_Path = new_FileDialog.getOpenFileName(filter= "Tensorflow checkpoint file (*.meta)")[0];
    if file_Path != "":
        ui.checkpoint_File_LineEdit.setText(os.path.splitext(file_Path)[0]);

def Next_Button_Event(hNet, window_Dict, ui):
    workspace_Path = ui.workspace_LineEdit.text();
    prerequisite_File_Path = ui.prerequisite_File_LineEdit.text();
    checkpoint_File_Path = ui.checkpoint_File_LineEdit.text();
    
    if workspace_Path == "":
        QtWidgets.QMessageBox.warning(None, 'Warning!', 'Set a workspace!');
        ui.workspace_LineEdit.setFocus();
        return;
    else:
        hNet.Set_Save_Path(workspace_Path)

    if prerequisite_File_Path == "":
        window_Dict["Process"].show();
    else:
        if os.path.splitext(prerequisite_File_Path)[1] == '.pickle':
            load_Function = hNet.Prerequisite_Load;
        elif os.path.splitext(prerequisite_File_Path)[1] == '.txt':
            load_Function = hNet.Prerequisite_Script_Load;
        else:
            raise ValueError('Unsupported prerequisite file extention.')
        response, e = load_Function(
            load_Prerequisite_File_Path= prerequisite_File_Path,
            load_Checkpoint_File_Path= checkpoint_File_Path if checkpoint_File_Path != "" else None
            )
        if not response:            
            QtWidgets.QMessageBox.critical(
                None,
                'Error!',
                'An error occurred while loading. '
                'Some states did not work properly. '
                'Please check the error message below. '
                '\n\n{}: {}'.format(type(e).__name__, e)
                )
            return;

        window_Dict["Train"].show();    
    
    window_Dict["Main"].close();