import sys, os;
from PyQt5 import QtGui, QtWidgets;
from HNet2_Core import HNet;
from GUI import Main, Process, Pattern, Learning, Train;

class HNet2_GUI:
    def __init__(self):
        self.hNet = HNet();

        self.Windows_Initialize();
        self.Event_Connect();

    def Windows_Initialize(self):
        self.windows_Dict = {
            "Main": QtWidgets.QMainWindow(),
            "Process": QtWidgets.QMainWindow(),
            "Pattern": QtWidgets.QMainWindow(),
            "Learning.Learning": QtWidgets.QMainWindow(),
            "Learning.Test": QtWidgets.QMainWindow(),
            "Train": QtWidgets.QMainWindow(),
            }

        self.ui_Dict = {
            "Main": Main.Form.Ui_MainWindow(),
            "Process": Process.Form.Ui_ProcessWindow(),
            "Pattern": Pattern.Form.Ui_PatternWindow(),
            "Learning.Learning": Learning.Learning.Form.Ui_LearningWindow(),
            "Learning.Test": Learning.Test.Form.Ui_TestWindow(),
            "Train": Train.Form.Ui_TrainWindow(),
            }

        self.ui_Dict["Main"].setupUi(self.windows_Dict["Main"])
        self.ui_Dict["Process"].setupUi(self.windows_Dict["Process"])
        self.ui_Dict["Pattern"].setupUi(self.windows_Dict["Pattern"])
        self.ui_Dict["Learning.Learning"].setupUi(self.windows_Dict["Learning.Learning"])
        self.ui_Dict["Learning.Test"].setupUi(self.windows_Dict["Learning.Test"])
        self.ui_Dict["Train"].setupUi(self.windows_Dict["Train"])

    def Event_Connect(self):
        Main.Events.Connect(self.hNet, self.windows_Dict, self.ui_Dict["Main"]);
        Process.Events.Connect(self.hNet, self.windows_Dict, self.ui_Dict["Process"]);
        Pattern.Events.Connect(self.hNet, self.windows_Dict, self.ui_Dict["Pattern"]);
        Learning.Learning.Events.Connect(self.hNet, self.windows_Dict, self.ui_Dict["Learning.Learning"]);
        Learning.Test.Events.Connect(self.hNet, self.windows_Dict, self.ui_Dict["Learning.Test"]);
        Train.Events.Connect(self.hNet, self.windows_Dict, self.ui_Dict["Train"]);

if __name__ == "__main__":    
    app = QtWidgets.QApplication(sys.argv)
    QtGui.QFontDatabase().addApplicationFont(os.path.join(os.getcwd(), 'D2Coding-Ver1.3.2-20180524.ttc').replace('\\', '/'))
    apply_Font = QtGui.QFont('D2Coding');
    apply_Font.setBold(False);
    apply_Font.setPointSize(9);
    app.setFont(apply_Font);
    hNet_GUI = HNet2_GUI();
    hNet_GUI.windows_Dict["Main"].show();
    
    sys.exit(app.exec_())
