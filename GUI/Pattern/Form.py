# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Pattern/Form.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_PatternWindow(object):
    def setupUi(self, PatternWindow):
        PatternWindow.setObjectName("PatternWindow")
        PatternWindow.resize(730, 800)
        self.centralwidget = QtWidgets.QWidget(PatternWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.preview_Tab_Widget = QtWidgets.QTabWidget(self.centralwidget)
        self.preview_Tab_Widget.setGeometry(QtCore.QRect(10, 180, 711, 561))
        self.preview_Tab_Widget.setObjectName("preview_Tab_Widget")
        self.add_PushButton = QtWidgets.QPushButton(self.centralwidget)
        self.add_PushButton.setGeometry(QtCore.QRect(580, 122, 141, 41))
        self.add_PushButton.setObjectName("add_PushButton")
        self.name_LineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.name_LineEdit.setGeometry(QtCore.QRect(10, 80, 113, 20))
        self.name_LineEdit.setObjectName("name_LineEdit")
        self.file_Label = QtWidgets.QLabel(self.centralwidget)
        self.file_Label.setGeometry(QtCore.QRect(10, 10, 81, 16))
        self.file_Label.setObjectName("file_Label")
        self.name_Label = QtWidgets.QLabel(self.centralwidget)
        self.name_Label.setGeometry(QtCore.QRect(10, 60, 81, 16))
        self.name_Label.setObjectName("name_Label")
        self.file_Browser_PushButton = QtWidgets.QPushButton(self.centralwidget)
        self.file_Browser_PushButton.setGeometry(QtCore.QRect(630, 30, 91, 23))
        self.file_Browser_PushButton.setObjectName("file_Browser_PushButton")
        self.file_LineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.file_LineEdit.setGeometry(QtCore.QRect(10, 30, 611, 20))
        self.file_LineEdit.setReadOnly(True)
        self.file_LineEdit.setObjectName("file_LineEdit")
        self.next_PushButton = QtWidgets.QPushButton(self.centralwidget)
        self.next_PushButton.setGeometry(QtCore.QRect(580, 750, 141, 41))
        self.next_PushButton.setObjectName("next_PushButton")
        self.notice_Label = QtWidgets.QLabel(self.centralwidget)
        self.notice_Label.setGeometry(QtCore.QRect(10, 740, 541, 16))
        self.notice_Label.setObjectName("notice_Label")
        PatternWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(PatternWindow)
        self.preview_Tab_Widget.setCurrentIndex(-1)
        QtCore.QMetaObject.connectSlotsByName(PatternWindow)

    def retranslateUi(self, PatternWindow):
        _translate = QtCore.QCoreApplication.translate
        PatternWindow.setWindowTitle(_translate("PatternWindow", "HNet 2.0 Alpha: Pattern"))
        self.add_PushButton.setText(_translate("PatternWindow", "Add"))
        self.file_Label.setText(_translate("PatternWindow", "Pattern file"))
        self.name_Label.setText(_translate("PatternWindow", "Name"))
        self.file_Browser_PushButton.setText(_translate("PatternWindow", "Browser"))
        self.next_PushButton.setText(_translate("PatternWindow", "Next"))
        self.notice_Label.setText(_translate("PatternWindow", "※ The preview only displays up to the 100th pattern."))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    PatternWindow = QtWidgets.QMainWindow()
    ui = Ui_PatternWindow()
    ui.setupUi(PatternWindow)
    PatternWindow.show()
    sys.exit(app.exec_())

