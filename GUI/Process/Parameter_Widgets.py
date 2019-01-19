from PyQt5 import QtCore, QtGui, QtWidgets;
import tensorflow as tf;
from HNet_Enum import Learning_Rate_Decay_Method;
from GUI.GUI_Enum import Parameter_Type;
from GUI import RegEx;

def Set_Widget(formLayoutWidget, formLayout, location, label_Text, parameter_Type, process_Name= None, tensor_List= None):    
    if parameter_Type == Parameter_Type.Name:
        return LineEdit_Widget(formLayoutWidget, formLayout, location, label_Text, parameter_Type).text;
    elif parameter_Type in [Parameter_Type.Int, Parameter_Type.Positive_int, Parameter_Type.Non_negative_int]:
        lineEdit_Widget = LineEdit_Widget(formLayoutWidget, formLayout, location, label_Text, parameter_Type)
        return lambda: int(lineEdit_Widget.text()) if lineEdit_Widget.text() != '' else None;
    elif parameter_Type in [Parameter_Type.Float, Parameter_Type.Positive_float]:
        lineEdit_Widget = LineEdit_Widget(formLayoutWidget, formLayout, location, label_Text, parameter_Type)
        return lambda: float(lineEdit_Widget.text()) if lineEdit_Widget.text() != '' else None;
    elif parameter_Type in [Parameter_Type.Int_list, Parameter_Type.Positive_int_list, Parameter_Type.Non_negative_int]:
        lineEdit_Widget = LineEdit_Widget(formLayoutWidget, formLayout, location, label_Text, parameter_Type);
        #return lambda: [int(x.strip()) if x != '' else None for x in lineEdit_Widget.text().split(',')];
        return lambda: [int(x.strip()) for x in lineEdit_Widget.text().split(',') if x != ''];
    elif parameter_Type in [Parameter_Type.Float_list, Parameter_Type.Positive_float_list]:
        lineEdit_Widget = LineEdit_Widget(formLayoutWidget, formLayout, location, label_Text, parameter_Type);
        return lambda: [float(x.strip()) if x != '' else None for x in lineEdit_Widget.text().split(',')];
    elif parameter_Type == Parameter_Type.Dtype:
        dtype_Widget = Dtype_Widget(formLayoutWidget, formLayout, location, label_Text);
        return lambda: dtype_Widget.itemData(dtype_Widget.currentIndex());
    elif parameter_Type == Parameter_Type.Bool:
        bool_Widget = Bool_Widget(formLayoutWidget, formLayout, location, label_Text);
        return lambda: bool_Widget.itemData(bool_Widget.currentIndex());
    elif parameter_Type == Parameter_Type.Padding:
        padding_Widget = Padding_Widget(formLayoutWidget, formLayout, location, label_Text);
        return lambda: padding_Widget.itemData(padding_Widget.currentIndex());
    elif parameter_Type == Parameter_Type.Decay_method:
        decay_Method_Widget = Decay_Method_Widget(formLayoutWidget, formLayout, location, label_Text);
        return lambda: decay_Method_Widget.itemData(decay_Method_Widget.currentIndex());
    elif parameter_Type == Parameter_Type.Tensor:
        tensor_Widget = Tensor_Widget(
            formLayoutWidget,
            formLayout,
            location,
            label_Text,
            tensor_List,
            use_Process_Name= process_Name,
            use_None= False
            )
        return lambda: tensor_Widget.itemData(tensor_Widget.currentIndex());
    elif parameter_Type == Parameter_Type.Tensor_with_None:
        tensor_Widget = Tensor_Widget(
            formLayoutWidget,
            formLayout,
            location,
            label_Text,
            tensor_List,
            use_Process_Name= process_Name,
            use_None= True
            )
        return lambda: tensor_Widget.itemData(tensor_Widget.currentIndex());
    elif parameter_Type == Parameter_Type.Tensor2:
        tensor_Widget1, tensor_Widget2 = Tensor2_Widget(
            formLayoutWidget,
            formLayout,
            location,
            label_Text,
            tensor_List,
            use_Process_Name= process_Name
            )
        return lambda: [
            tensor_Widget1.itemData(tensor_Widget1.currentIndex()),
            tensor_Widget2.itemData(tensor_Widget2.currentIndex())
            ]
    elif parameter_Type == Parameter_Type.Tensor_list:
        tensor_List_Widget = Tensor_List_Widget(
            formLayoutWidget,
            formLayout,
            location,
            label_Text,
            tensor_List,
            use_Process_Name= process_Name
            )
        return lambda: [tensor_List_Widget.item(row).data(QtCore.Qt.UserRole) for row in range(tensor_List_Widget.count())];
    elif parameter_Type == Parameter_Type.Process_and_Tensor:
        tensor_Widget = Tensor_Widget(
            formLayoutWidget,
            formLayout,
            location,
            label_Text,
            tensor_List,
            use_Process_Name= None,
            use_None= False
            )
        return lambda: tensor_Widget.itemData(tensor_Widget.currentIndex());
    elif parameter_Type == Parameter_Type.Process_and_Tensor_with_None:
        tensor_Widget = Tensor_Widget(
            formLayoutWidget,
            formLayout,
            location,
            label_Text,
            tensor_List,
            use_Process_Name= None,
            use_None= True
            )
        return lambda: tensor_Widget.itemData(tensor_Widget.currentIndex());
    elif parameter_Type == Parameter_Type.Process_and_Tensor2:
        tensor_Widget1, tensor_Widget2 = Tensor2_Widget(
            formLayoutWidget,
            formLayout,
            location,
            label_Text,
            tensor_List,
            use_Process_Name= None
            )
        return lambda: [
            tensor_Widget1.itemData(tensor_Widget1.currentIndex()),
            tensor_Widget2.itemData(tensor_Widget2.currentIndex())
            ]
    elif parameter_Type == Parameter_Type.Process_and_Tensor_list:
        tensor_List_Widget = Tensor_List_Widget(
            formLayoutWidget,
            formLayout,
            location,
            label_Text,
            tensor_List,
            use_Process_Name= None
            )
        return lambda: [tensor_List_Widget.item(row).data(QtCore.Qt.UserRole) for row in range(tensor_List_Widget.count())];
    elif parameter_Type == Parameter_Type.Variable_list:
        variable_List_Widget = Variable_List_Widget(
            formLayoutWidget,
            formLayout,
            location,
            label_Text,
            use_Process_Name= process_Name
            )
        return lambda: [variable_List_Widget.item(row).data(QtCore.Qt.UserRole) for row in range(variable_List_Widget.count())];
    elif parameter_Type == Parameter_Type.Process_and_Variable_list:
        variable_List_Widget = Variable_List_Widget(
            formLayoutWidget,
            formLayout,
            location,
            label_Text,
            use_Process_Name= None
            )
        return lambda: [variable_List_Widget.item(row).data(QtCore.Qt.UserRole) for row in range(variable_List_Widget.count())];

    elif parameter_Type == Parameter_Type.Hidden_activation_func:
        activaiton_Func_Widget = Hidden_Activation_Func_Widget(formLayoutWidget, formLayout, location, label_Text);
        return lambda: activaiton_Func_Widget.itemData(activaiton_Func_Widget.currentIndex());
    elif parameter_Type == Parameter_Type.Output_activation_func:
        activaiton_Func_Widget = Output_Activation_Func_Widget(formLayoutWidget, formLayout, location, label_Text);
        return lambda: activaiton_Func_Widget.itemData(activaiton_Func_Widget.currentIndex());
    elif parameter_Type == Parameter_Type.RNN_cell_func:
        rnn_Cell_Func_Widget = RNN_Cell_Func_Widget(formLayoutWidget, formLayout, location, label_Text);
        return lambda: rnn_Cell_Func_Widget.itemData(rnn_Cell_Func_Widget.currentIndex());
    else:
        raise ValueError("Unknown parameter type");

def Form_Layout(container_Tab):
    new_FormLayoutWidget = QtWidgets.QWidget(container_Tab)
    new_FormLayoutWidget.setGeometry(QtCore.QRect(330, 290, 481, 261))    
    new_FormLayout = QtWidgets.QFormLayout(new_FormLayoutWidget)
    new_FormLayout.setContentsMargins(0, 0, 0, 0)

    return new_FormLayoutWidget, new_FormLayout;

def LineEdit_Widget(formLayoutWidget, formLayout, location, label_Text, parameter_Type):
    new_Label = QtWidgets.QLabel(formLayoutWidget)
    formLayout.setWidget(location, QtWidgets.QFormLayout.LabelRole, new_Label);
    new_Label.setText(label_Text)

    new_LineEdit = QtWidgets.QLineEdit(formLayoutWidget)
    formLayout.setWidget(location, QtWidgets.QFormLayout.FieldRole, new_LineEdit);

    if parameter_Type == Parameter_Type.Name:
        new_LineEdit.setValidator(RegEx.Letter)
    elif parameter_Type == Parameter_Type.Int:
        new_LineEdit.setValidator(RegEx.Int)
    elif parameter_Type == Parameter_Type.Positive_int:
        new_LineEdit.setValidator(RegEx.Positive_Int)
    elif parameter_Type == Parameter_Type.Int_list:
        new_LineEdit.setValidator(RegEx.Int_List)
    elif parameter_Type == Parameter_Type.Positive_int_list:
        new_LineEdit.setValidator(RegEx.Positive_Int_List)
    elif parameter_Type == Parameter_Type.Float:
        new_LineEdit.setValidator(RegEx.Float)
    elif parameter_Type == Parameter_Type.Positive_float:
        new_LineEdit.setValidator(RegEx.Positive_Float)
    elif parameter_Type == Parameter_Type.Float_list:
        new_LineEdit.setValidator(RegEx.Float_List)
    elif parameter_Type == Parameter_Type.Positive_float_list:
        new_LineEdit.setValidator(RegEx.Positive_Float_List)

    elif parameter_Type == Parameter_Type.Non_negative_int:
        new_LineEdit.setValidator(RegEx.Non_Negative_Int)
    elif parameter_Type == Parameter_Type.Non_negative_int_list:
        new_LineEdit.setValidator(RegEx.Non_Negative_Int_List)

    return new_LineEdit;

def Combobox_Widget(formLayoutWidget, formLayout, location, label_Text, item_List):
    '''
    item_List: tuple of (str, obj)
    '''
    new_Label = QtWidgets.QLabel(formLayoutWidget)
    formLayout.setWidget(location, QtWidgets.QFormLayout.LabelRole, new_Label);
    new_Label.setText(label_Text)

    new_ComboBox = QtWidgets.QComboBox(formLayoutWidget);
    formLayout.setWidget(location, QtWidgets.QFormLayout.FieldRole, new_ComboBox);
    for item_Text, item_Obj in item_List:
        new_ComboBox.addItem(item_Text, QtCore.QVariant(item_Obj))

    return new_ComboBox;

def Dtype_Widget(formLayoutWidget, formLayout, location, label_Text):
    item_List = [
        ("Float32", tf.float32),
        ("Int32", tf.int32),
        ("Bool", tf.bool),          
        ]
    return Combobox_Widget(formLayoutWidget, formLayout, location, label_Text, item_List)

def Bool_Widget(formLayoutWidget, formLayout, location, label_Text):
    item_List = [
        ("True", True),
        ("False", False),
        ]
    return Combobox_Widget(formLayoutWidget, formLayout, location, label_Text, item_List)

def Padding_Widget(formLayoutWidget, formLayout, location, label_Text):
    item_List = [
        ("SAME", "same"),
        ("VALID", "valid"),
        ]
    return Combobox_Widget(formLayoutWidget, formLayout, location, label_Text, item_List)

def Decay_Method_Widget(formLayoutWidget, formLayout, location, label_Text):
    item_List = [
        ("No Decay", Learning_Rate_Decay_Method.No_Decay),
        ("Exponential", Learning_Rate_Decay_Method.Exponential),
        ("Noam", Learning_Rate_Decay_Method.Noam),
        ]
    return Combobox_Widget(formLayoutWidget, formLayout, location, label_Text, item_List)

def Tensor_Widget(formLayoutWidget, formLayout, location, label_Text, tensor_List, use_Process_Name = None, use_None = False):
    '''
    item_List: tuple of (str, obj)
    '''
    if not use_Process_Name is None:
        tensor_List = [tensor_Name for process_Name, tensor_Name in tensor_List if process_Name == use_Process_Name];
    if use_None:
        tensor_List = [None] + tensor_List; 

    new_Label = QtWidgets.QLabel(formLayoutWidget)
    formLayout.setWidget(location, QtWidgets.QFormLayout.LabelRole, new_Label);
    new_Label.setText(label_Text)

    new_ComboBox = QtWidgets.QComboBox(formLayoutWidget);
    formLayout.setWidget(location, QtWidgets.QFormLayout.FieldRole, new_ComboBox);
    for item in tensor_List:        
        new_ComboBox.addItem('{}'.format(item), item);    
    new_ComboBox.setCurrentIndex(new_ComboBox.count() - 1);
    if use_None:
        new_ComboBox.setCurrentIndex(0);

    return new_ComboBox;

def Tensor2_Widget(formLayoutWidget, formLayout, location, label_Text, tensor_List, use_Process_Name = None):
    '''
    tensor_List: a list of str(tensor name)
    '''
    if not use_Process_Name is None:
        tensor_List = [tensor_Name for process_Name, tensor_Name in tensor_List if process_Name == use_Process_Name];

    new_Label = QtWidgets.QLabel(formLayoutWidget)
    formLayout.setWidget(location, QtWidgets.QFormLayout.LabelRole, new_Label);
    new_Label.setText(label_Text)
        
    new_VerticalLayout = QtWidgets.QHBoxLayout()
    new_VerticalLayout.setContentsMargins(0, 0, 0, 0)

    new_ComboBox1 = QtWidgets.QComboBox(formLayoutWidget);
    new_VerticalLayout.addWidget(new_ComboBox1);
    for item in tensor_List:
        new_ComboBox1.addItem('{}'.format(item), item);
    new_ComboBox1.setCurrentIndex(new_ComboBox1.count() - 1);

    new_ComboBox2 = QtWidgets.QComboBox(formLayoutWidget);
    new_VerticalLayout.addWidget(new_ComboBox2);
    for item in tensor_List:
        new_ComboBox2.addItem('{}'.format(item), item);
    new_ComboBox2.setCurrentIndex(new_ComboBox2.count() - 1);

    formLayout.setLayout(location, QtWidgets.QFormLayout.FieldRole, new_VerticalLayout);
    
    return new_ComboBox1, new_ComboBox2

def Tensor_List_Widget(formLayoutWidget, formLayout, location, label_Text, tensor_List, use_Process_Name = None):
    '''
    tensor_List: tuple of (str, obj)
    '''
    if not use_Process_Name is None:
        tensor_List = [tensor_Name for process_Name, tensor_Name in tensor_List if process_Name == use_Process_Name];

    new_Label = QtWidgets.QLabel(formLayoutWidget)
    formLayout.setWidget(location, QtWidgets.QFormLayout.LabelRole, new_Label);
    new_Label.setText(label_Text)

    new_GridLayout = QtWidgets.QGridLayout()
    new_GridLayout.setContentsMargins(0, 0, 0, 0)

    new_ListWidget = QtWidgets.QListWidget(formLayoutWidget)    
    new_GridLayout.addWidget(new_ListWidget, 0, 0, 2, 1);
    
    new_ComboBox = QtWidgets.QComboBox(formLayoutWidget);
    new_GridLayout.addWidget(new_ComboBox, 0, 1, 1, 1);
    for item in tensor_List:        
        new_ComboBox.addItem('{}'.format(item), item);
    new_ComboBox.setCurrentIndex(new_ComboBox.count() - 1);

    new_Add_Button = QtWidgets.QPushButton(formLayoutWidget);
    new_Add_Button.setText("Add");
    new_GridLayout.addWidget(new_Add_Button, 1, 1, 1, 1);

    new_GridLayout.setColumnStretch(0, 1)
    new_GridLayout.setColumnStretch(1, 1)

    formLayout.setLayout(location, QtWidgets.QFormLayout.FieldRole, new_GridLayout);

    def Add_Event():
        if new_ComboBox.currentIndex == 0:
            new_ComboBox.setFocus();
            return;
        
        new_Item = QtWidgets.QListWidgetItem(new_ComboBox.currentText(), new_ListWidget);
        new_Item.setData(QtCore.Qt.UserRole, new_ComboBox.itemData(new_ComboBox.currentIndex()));
        new_ListWidget.addItem(new_Item);

    def Delete_Event():
        if new_ListWidget.currentRow() < 0:
            new_ListWidget.setFocus();
            return;

        new_ListWidget.takeItem(new_ListWidget.currentRow()).data(QtCore.Qt.UserRole)

    new_Add_Button.clicked.connect(Add_Event);
    new_ListWidget.doubleClicked.connect(Delete_Event);

    return new_ListWidget

def Variable_List_Widget(formLayoutWidget, formLayout, location, label_Text, use_Process_Name = None):
    '''
    This is using 'tf.trainable_variables()
    '''
    variable_List = [x.name for x in tf.trainable_variables()];
    if not use_Process_Name is None:
        variable_List = [x.name for x in tf.trainable_variables() if x.name.startswith(use_Process_Name)];    

    new_Label = QtWidgets.QLabel(formLayoutWidget)
    formLayout.setWidget(location, QtWidgets.QFormLayout.LabelRole, new_Label);
    new_Label.setText(label_Text)

    new_GridLayout = QtWidgets.QGridLayout()
    new_GridLayout.setContentsMargins(0, 0, 0, 0)

    new_ListWidget = QtWidgets.QListWidget(formLayoutWidget)    
    new_GridLayout.addWidget(new_ListWidget, 0, 0, 2, 1);
    
    new_ComboBox = QtWidgets.QComboBox(formLayoutWidget);
    new_GridLayout.addWidget(new_ComboBox, 0, 1, 1, 1);

    for item in variable_List:
        item_Text = item;
        if not use_Process_Name is None:
            item_Text = item_Text.replace('{}/'.format(use_Process_Name), '');
        new_Item = QtWidgets.QListWidgetItem(item_Text, new_ListWidget);
        new_Item.setData(QtCore.Qt.UserRole, item);
        new_ListWidget.addItem(new_Item);
        
    new_Add_Button = QtWidgets.QPushButton(formLayoutWidget);
    new_Add_Button.setText("Add");
    new_GridLayout.addWidget(new_Add_Button, 1, 1, 1, 1);

    new_GridLayout.setColumnStretch(0, 1)
    new_GridLayout.setColumnStretch(1, 1)

    formLayout.setLayout(location, QtWidgets.QFormLayout.FieldRole, new_GridLayout);

    def Add_Event():
        if new_ComboBox.currentIndex == 0:
            new_ComboBox.setFocus();
            return;
        
        new_Item = QtWidgets.QListWidgetItem(new_ComboBox.currentText(), new_ListWidget);
        new_Item.setData(QtCore.Qt.UserRole, new_ComboBox.itemData(new_ComboBox.currentIndex()));
        new_ListWidget.addItem(new_Item);

    def Delete_Event():
        if new_ListWidget.currentRow() < 0:
            new_ListWidget.setFocus();
            return;

        new_ListWidget.takeItem(new_ListWidget.currentRow()).data(QtCore.Qt.UserRole)
        
    #Combobox_Item reassign
    def List_Change_Event():
        new_ComboBox.clear();
        for item in variable_List:
            if not item in [new_ListWidget.item(row).data(QtCore.Qt.UserRole) for row in range(new_ListWidget.count())]:
                item_Text = item;
                if not use_Process_Name is None:
                    item_Text = item_Text.replace('{}/'.format(use_Process_Name), '');
                new_ComboBox.addItem('{}'.format(item_Text), item);

    new_Add_Button.clicked.connect(Add_Event);
    new_Add_Button.clicked.connect(List_Change_Event);
    new_ListWidget.doubleClicked.connect(Delete_Event);
    new_ListWidget.doubleClicked.connect(List_Change_Event);

    return new_ListWidget


def Hidden_Activation_Func_Widget(formLayoutWidget, formLayout, location, label_Text):
    item_List = [
        ("Sigmoid", tf.nn.sigmoid),
        ("Tanh", tf.nn.tanh),
        ("ReLU", tf.nn.relu),
        ("Leaky_ReLU", tf.nn.leaky_relu),
        ("Softplus", tf.nn.softplus),           
        ]
    return Combobox_Widget(formLayoutWidget, formLayout, location, label_Text, item_List)

def Output_Activation_Func_Widget(formLayoutWidget, formLayout, location, label_Text):
    item_List = [
        ("Sigmoid", tf.nn.sigmoid),
        ("Softmax", tf.nn.softmax)
        ]
    return Combobox_Widget(formLayoutWidget, formLayout, location, label_Text, item_List)

def RNN_Cell_Func_Widget(formLayoutWidget, formLayout, location, label_Text):
    item_List = [
        ("LSTM", (tf.nn.rnn_cell.LSTMCell, tf.nn.rnn_cell.LSTMStateTuple)),
        ("GRU", (tf.nn.rnn_cell.GRUCell, None)),
        ("Basic", (tf.nn.rnn_cell.BasicRNNCell, None)),
        ]
    return Combobox_Widget(formLayoutWidget, formLayout, location, label_Text, item_List)