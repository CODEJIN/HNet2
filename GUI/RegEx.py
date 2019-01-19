from PyQt5 import QtCore, QtGui;

Float = QtGui.QRegExpValidator(QtCore.QRegExp("[-+]?([0-9]+\.?[0-9]*|\.[0-9]+)([eE][-+]?[0-9]+)?"));
Float_List = QtGui.QRegExpValidator(QtCore.QRegExp("([-+]?([0-9]+\.?[0-9]*|\.[0-9]+)([eE][-+]?[0-9]+)?,\s?)*"));
Positive_Float = QtGui.QRegExpValidator(QtCore.QRegExp("([0-9]+\.?[0-9]*|\.[0-9]+)([eE][-+]?[0-9]+)?"));
Positive_Float_List = QtGui.QRegExpValidator(QtCore.QRegExp("(([0-9]+\.?[0-9]*|\.[0-9]+)([eE][-+]?[0-9]+)?,\s?)*"));
Int = QtGui.QRegExpValidator(QtCore.QRegExp("[-+]?([0-9]+)?"));
Int_List = QtGui.QRegExpValidator(QtCore.QRegExp("([-+]?([0-9]+)+,\s?)*"));
Positive_Int = QtGui.QRegExpValidator(QtCore.QRegExp("^[1-9][0-9]*$"));
Non_Negative_Int = QtGui.QRegExpValidator(QtCore.QRegExp("[0-9]*$"));
Positive_Int_List = QtGui.QRegExpValidator(QtCore.QRegExp("([1-9][0-9]*,\s?)*"));
Non_Negative_Int_List = QtGui.QRegExpValidator(QtCore.QRegExp("([0-9]+,\s?)*"));
Letter = QtGui.QRegExpValidator(QtCore.QRegExp("^[0-9A-Za-z_]+$"));

