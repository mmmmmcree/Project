"""
pyside2-rcc resource.qrc -o resource.py
pyside2-uic Login.ui > Login.py
"""
import time

from PySide2 import QtCore, QtWidgets
from PySide2.QtCore import Qt, QPropertyAnimation, QRect, QEasingCurve, QObject, Signal
from PySide2.QtGui import QColor, QIcon
from PySide2.QtUiTools import QUiLoader
from PySide2.QtWidgets import QApplication, QMainWindow, QWidget, QDesktopWidget, QTextBrowser, QInputDialog
from Login import Ui_Form
from threading import Timer

from UI设计.LIB.share import Shared_Info
from UI设计.MainWindow import Ui_MainWindow
from UI设计.新建文本文档副本 import Win_Main, Win_Login, Parking_Login, Face_Login


class MySignals(QObject):
    text_print = Signal(QTextBrowser, str)


class Win_Choice:

    def __init__(self):
        self.ms = MySignals()
        self.ui = QUiLoader().load('Choice.ui')
        self.ui.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        self.ui.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.ui.pushButton_2.setEnabled(False)
        self.ui.pushButton.clicked.connect(self.ui.close)
        self.ui.pushButton_2.clicked.connect(self.enter)
        self.ui.listWidget.itemSelectionChanged.connect(self.handleSelectionChange)
        self.ms.text_print.connect(self.printTextToGui)

    def printTextToGui(self, widget, text):
        widget.append(str(text))
        widget.ensureCursorVisible()

    def handleSelectionChange(self):
        self.choice = self.ui.listWidget.currentItem().text()
        self.ui.pushButton_2.setEnabled(True)
        with open(f'texts/{self.choice}.txt', 'r') as f:
            text = f.read()
        self.ui.textBrowser.clear()
        self.ms.text_print.emit(self.ui.textBrowser, text)

    def enter(self):
        if self.choice == '停车场管理':
            Shared_Info.MainWindow = Parking_Login()
            Shared_Info.MainWindow.ui.show()
            self.ui.close()
        elif self.choice == '基础功能演示':
            Shared_Info.MainWindow = Win_Login()
            Shared_Info.MainWindow.ui.show()
            self.ui.close()
        elif self.choice == '人脸识别':
            Shared_Info.MainWindow = Face_Login()
            Shared_Info.MainWindow.ui.show()
            self.ui.close()


def LOGIN_UI_SHOW():
    app = QApplication([])
    app.setWindowIcon(QIcon('images/logo.png'))
    Shared_Info.Login = Win_Choice()
    Shared_Info.Login.ui.show()
    app.exec_()


if __name__ == "__main__":
    LOGIN_UI_SHOW()
