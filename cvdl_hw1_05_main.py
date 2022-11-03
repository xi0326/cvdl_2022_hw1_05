from PyQt5 import QtWidgets, QtCore, QtGui
import sys
import MainWindow as ui
import os

from Q5.Q5 import Question5

class Main(QtWidgets.QMainWindow, ui.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.imagePath = None

        # load data
        self.pushButtonLoadImage.clicked.connect(self.getImagePath)

        # question
        self.pushButtonShowTrainImages.clicked.connect(Q5Object.showTrainImages)
        self.pushButtonShowDataAugmentation.clicked.connect(lambda: Q5Object.showDataAugmentation(self.imagePath))
        self.pushButtonShowModelStructure.clicked.connect(Q5Object.showModelStructure)
        self.pushButtonShowAccuracyAndLoss.clicked.connect(self.showAccuracyAndLoss)
        self.pushButtonShowInference.clicked.connect(lambda: self.showInference(self.imagePath))


    
    def selectFile(self):
        fileName = QtCore.QDir.toNativeSeparators(QtWidgets.QFileDialog.getOpenFileName(None, caption='Choose a File', directory='C:\\', filter='Image Files (*.png *.jpg *.bmp)')[0])  # get turple[0] which is file name
        return fileName
    
    def getImagePath(self):
        self.imagePath = self.selectFile()
        self.photo.setPixmap(QtGui.QPixmap(self.imagePath)) # show the image on the UI

    def showAccuracyAndLoss(self):
        Q5Object.makeAccuracyAndLoss()
        self.photo.setPixmap(QtGui.QPixmap('Q5/result.png'))

    def showInference(self, imgPath):
        conf, label = Q5Object.showInference(imgPath)
        self.textArea.setText('Confidence = ' + str(conf) + '\n' + 'Prediction Label: ' + label)

    # overide to force exit
    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        super().closeEvent(a0)
        os._exit(0)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    Q5Object = Question5()
    window = Main()
    window.show()
    sys.exit(app.exec_())