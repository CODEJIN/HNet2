from PyQt5 import QtCore, QtGui, QtWidgets;
import sys, os, importlib;
from GUI.Process import Parameter_Widgets;

def Generate_Tab(hNet, ui, module_Name, graphic_File = None):
    imported_Module = importlib.import_module(module_Name)

    new_Shortcut_Tab = QtWidgets.QWidget()
    ui.tesnor_Tab_Widget.insertTab(ui.tesnor_Tab_Widget.count() - 1, new_Shortcut_Tab, module_Name);
        
    new_Shortcut_Graphics_Label = QtWidgets.QLabel(new_Shortcut_Tab)
    new_Shortcut_Graphics_Label.setGeometry(QtCore.QRect(10, 10, 150, 12))
    new_Shortcut_Graphics_Label.setText('Shortcut example')
    new_ShortcutGraphicsView_Label = QtWidgets.QLabel(new_Shortcut_Tab)
    new_ShortcutGraphicsView_Label.setGeometry(QtCore.QRect(10, 30, 301, 641))
    new_Shortcut_Description_Label = QtWidgets.QLabel(new_Shortcut_Tab)
    new_Shortcut_Description_Label.setGeometry(QtCore.QRect(330, 10, 150, 12))
    new_Shortcut_Description_Label.setText('Shortcut description')
    new_Shortcut_Description_PlainTextEdit = QtWidgets.QPlainTextEdit(new_Shortcut_Tab)
    new_Shortcut_Description_PlainTextEdit.setGeometry(QtCore.QRect(330, 30, 481, 201))
    new_Shortcut_Description_PlainTextEdit.setReadOnly(True)
    new_Shortcut_Parameter_Setup_Label = QtWidgets.QLabel(new_Shortcut_Tab)
    new_Shortcut_Parameter_Setup_Label.setGeometry(QtCore.QRect(330, 270, 150, 12))
    new_Shortcut_Parameter_Setup_Label.setText('Parameter setup')
    new_FormLayoutWidget, new_FormLayout = Parameter_Widgets.Form_Layout(new_Shortcut_Tab);
    new_FormLayoutWidget.setGeometry(QtCore.QRect(330, 290, 481, 321))
    new_Add_PushButton = QtWidgets.QPushButton(new_Shortcut_Tab)        
    new_Add_PushButton.setGeometry(QtCore.QRect(490, 620, 171, 41))
    new_Add_PushButton.setText('Add');

    new_Shortcut_Description_PlainTextEdit.setPlainText(imported_Module.description.strip());

    new_Parameter_Func_Dict = {};        
    for row_Index, (parameter_Name, parameter_Type) in enumerate(imported_Module.parameter_Dict.items()):
        new_Parameter_Func_Dict[parameter_Name] = Parameter_Widgets.Set_Widget(
            formLayoutWidget= new_FormLayoutWidget,
            formLayout= new_FormLayout,
            location= row_Index,
            label_Text= parameter_Name,
            parameter_Type= parameter_Type
            )
    if not graphic_File is None:
        new_Pixmap = QtGui.QPixmap(graphic_File);
        if new_Pixmap.width() / 300 > new_Pixmap.height() / 640:
            new_Pixmap = new_Pixmap.scaledToWidth(300, QtCore.Qt.SmoothTransformation);
        else:
            new_Pixmap = new_Pixmap.scaledToHeight(640, QtCore.Qt.SmoothTransformation);
        new_ShortcutGraphicsView_Label.setPixmap(new_Pixmap);
        new_ShortcutGraphicsView_Label.setAlignment(QtCore.Qt.AlignCenter);

    def Generate_Parameter_Dict(parameter_Func_Dict):
        parameter_Dict = {'process_Name': ui.process_ListWidget.currentItem().text()};
        parameter_Dict.update({key: value() for key, value in parameter_Func_Dict.items()});
        return parameter_Dict;

    def Clear():
        for index in range(new_FormLayout.count()):
            widget = new_FormLayout.itemAt(index).widget();
            if isinstance(widget, QtWidgets.QLineEdit):
                widget.clear();
            elif isinstance(widget, QtWidgets.QComboBox):
                widget.setCurrentIndex(0);
            elif isinstance(widget, QtWidgets.QListView):
                widget.clear();

    new_Add_PushButton.clicked.connect(lambda: imported_Module.Run_Shortcut(hNet, **Generate_Parameter_Dict(new_Parameter_Func_Dict)))        
    new_Add_PushButton.clicked.connect(Clear);
    
    return new_Shortcut_Tab, new_Add_PushButton;