# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'MainWindow.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

import resource

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(663, 623)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.label = QLabel(self.centralwidget)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(400, 210, 191, 101))
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(30)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setStyleSheet(u"border:2px solid;\n"
"border-radius:30;\n"
"background-color: rgba(255, 255, 255, 0);")
        self.pushButton = QPushButton(self.centralwidget)
        self.pushButton.setObjectName(u"pushButton")
        self.pushButton.setGeometry(QRect(40, 470, 256, 25))
        font = QFont()
        font.setFamily(u"Microsoft YaHei UI")
        font.setPointSize(10)
        self.pushButton.setFont(font)
        self.pushButton.setStyleSheet(u"border-image: url(:/images/images/\u6309\u94ae.png);")
        self.textBrowser = QTextBrowser(self.centralwidget)
        self.textBrowser.setObjectName(u"textBrowser")
        self.textBrowser.setGeometry(QRect(10, 60, 256, 192))
        self.textBrowser.setStyleSheet(u"border:2px solid;\n"
"border-radius:30;\n"
"background-color: rgba(255, 255, 255, 0);\n"
"")
        self.pushButton_2 = QPushButton(self.centralwidget)
        self.pushButton_2.setObjectName(u"pushButton_2")
        self.pushButton_2.setGeometry(QRect(50, 400, 75, 23))
        self.label_2 = QLabel(self.centralwidget)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setGeometry(QRect(390, 350, 256, 126))
        sizePolicy.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy)
        self.label_2.setStyleSheet(u"border:2px solid;\n"
"border-radius:30;\n"
"background-color: rgba(255, 255, 255, 0);")
        self.label_3 = QLabel(self.centralwidget)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setGeometry(QRect(0, 40, 661, 491))
        self.label_3.setStyleSheet(u"border-image: url(:/images/images/bkk.png);\n"
"border-radius:50")
        MainWindow.setCentralWidget(self.centralwidget)
        self.label_3.raise_()
        self.label.raise_()
        self.pushButton.raise_()
        self.textBrowser.raise_()
        self.pushButton_2.raise_()
        self.label_2.raise_()
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 663, 22))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"\u5c55\u793a\u8f66\u724c", None))
        self.pushButton.setText(QCoreApplication.translate("MainWindow", u"\u8bc6\u522b", None))
        self.pushButton_2.setText(QCoreApplication.translate("MainWindow", u"\u8fd4\u56de", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"\u5c55\u793a\u539f\u56fe", None))
        self.label_3.setText("")
    # retranslateUi