# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Main/Form.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(609, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.title_TextEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.title_TextEdit.setGeometry(QtCore.QRect(10, 10, 591, 281))
        font = QtGui.QFont()
        font.setFamily("D2Coding")
        self.title_TextEdit.setFont(font)
        self.title_TextEdit.setReadOnly(True)
        self.title_TextEdit.setObjectName("title_TextEdit")
        self.workspace_Label = QtWidgets.QLabel(self.centralwidget)
        self.workspace_Label.setGeometry(QtCore.QRect(20, 310, 150, 16))
        self.workspace_Label.setObjectName("workspace_Label")
        self.workspace_LineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.workspace_LineEdit.setGeometry(QtCore.QRect(20, 330, 471, 20))
        self.workspace_LineEdit.setReadOnly(True)
        self.workspace_LineEdit.setObjectName("workspace_LineEdit")
        self.next_PushButton = QtWidgets.QPushButton(self.centralwidget)
        self.next_PushButton.setGeometry(QtCore.QRect(454, 520, 141, 41))
        self.next_PushButton.setObjectName("next_PushButton")
        self.workspace_Browser_PushButton = QtWidgets.QPushButton(self.centralwidget)
        self.workspace_Browser_PushButton.setGeometry(QtCore.QRect(504, 330, 91, 23))
        self.workspace_Browser_PushButton.setObjectName("workspace_Browser_PushButton")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(10, 380, 591, 121))
        self.groupBox.setObjectName("groupBox")
        self.prerequisite_File_Label = QtWidgets.QLabel(self.groupBox)
        self.prerequisite_File_Label.setGeometry(QtCore.QRect(10, 20, 150, 16))
        self.prerequisite_File_Label.setObjectName("prerequisite_File_Label")
        self.prerequisite_File_LineEdit = QtWidgets.QLineEdit(self.groupBox)
        self.prerequisite_File_LineEdit.setGeometry(QtCore.QRect(10, 40, 471, 20))
        self.prerequisite_File_LineEdit.setReadOnly(True)
        self.prerequisite_File_LineEdit.setObjectName("prerequisite_File_LineEdit")
        self.checkpoint_File_LineEdit = QtWidgets.QLineEdit(self.groupBox)
        self.checkpoint_File_LineEdit.setGeometry(QtCore.QRect(10, 90, 471, 20))
        self.checkpoint_File_LineEdit.setReadOnly(True)
        self.checkpoint_File_LineEdit.setObjectName("checkpoint_File_LineEdit")
        self.checkpoint_File_Label = QtWidgets.QLabel(self.groupBox)
        self.checkpoint_File_Label.setGeometry(QtCore.QRect(10, 70, 150, 16))
        self.checkpoint_File_Label.setObjectName("checkpoint_File_Label")
        self.prerequisite_File_Browser_PushButton = QtWidgets.QPushButton(self.groupBox)
        self.prerequisite_File_Browser_PushButton.setGeometry(QtCore.QRect(494, 40, 91, 23))
        self.prerequisite_File_Browser_PushButton.setObjectName("prerequisite_File_Browser_PushButton")
        self.checkpoint_File_Browser_PushButton = QtWidgets.QPushButton(self.groupBox)
        self.checkpoint_File_Browser_PushButton.setGeometry(QtCore.QRect(494, 90, 91, 23))
        self.checkpoint_File_Browser_PushButton.setObjectName("checkpoint_File_Browser_PushButton")
        self.device_Signal_Label = QtWidgets.QLabel(self.centralwidget)
        self.device_Signal_Label.setGeometry(QtCore.QRect(10, 570, 20, 16))
        self.device_Signal_Label.setObjectName("device_Signal_Label")
        self.device_Info_Label = QtWidgets.QLabel(self.centralwidget)
        self.device_Info_Label.setGeometry(QtCore.QRect(40, 570, 551, 16))
        self.device_Info_Label.setText("")
        self.device_Info_Label.setObjectName("device_Info_Label")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "HNet 2.0 Alpha"))
        self.title_TextEdit.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'D2Coding\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'Gulim\'; font-size:24pt;\">HNet 2.0 Alpha ver.</span></p>\n"
"<p align=\"center\" style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-family:\'Gulim\';\"><br /></p>\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'Gulim\';\">하아....</span></p>\n"
"<p align=\"center\" style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-family:\'Gulim\';\"><br /></p>\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'Gulim\';\">2019 - {} Heejo You reserved.</span></p>\n"
"<p align=\"center\" style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-family:\'Gulim\';\"><br /></p></body></html>"))
        self.workspace_Label.setText(_translate("MainWindow", "Workspace"))
        self.next_PushButton.setText(_translate("MainWindow", "Next"))
        self.workspace_Browser_PushButton.setText(_translate("MainWindow", "Browser"))
        self.groupBox.setTitle(_translate("MainWindow", "Prerequisite load"))
        self.prerequisite_File_Label.setText(_translate("MainWindow", "Prerequisite file"))
        self.checkpoint_File_Label.setText(_translate("MainWindow", "Checkpoint file"))
        self.prerequisite_File_Browser_PushButton.setText(_translate("MainWindow", "Browser"))
        self.checkpoint_File_Browser_PushButton.setText(_translate("MainWindow", "Browser"))
        self.device_Signal_Label.setText(_translate("MainWindow", ".●"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

