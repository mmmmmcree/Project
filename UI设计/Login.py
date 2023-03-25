# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'Login.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

import resource

class Ui_Form(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(650, 750)
        Form.setMinimumSize(QSize(650, 750))
        Form.setMaximumSize(QSize(650, 900))
        self.radioButton = QRadioButton(Form)
        self.buttonGroup = QButtonGroup(Form)
        self.buttonGroup.setObjectName(u"buttonGroup")
        self.buttonGroup.addButton(self.radioButton)
        self.radioButton.setObjectName(u"radioButton")
        self.radioButton.setGeometry(QRect(300, 320, 151, 61))
        font = QFont()
        font.setFamily(u"Microsoft YaHei UI")
        self.radioButton.setFont(font)
        icon = QIcon()
        icon.addFile(u"../../yolov5-7.0/UI\u8bbe\u8ba1/images/\u6444\u50cf\u5934_camera-one.png", QSize(), QIcon.Normal, QIcon.Off)
        self.radioButton.setIcon(icon)
        self.radioButton_2 = QRadioButton(Form)
        self.buttonGroup.addButton(self.radioButton_2)
        self.radioButton_2.setObjectName(u"radioButton_2")
        self.radioButton_2.setGeometry(QRect(110, 310, 161, 61))
        font1 = QFont()
        font1.setFamily(u"Microsoft YaHei UI")
        font1.setBold(False)
        font1.setWeight(50)
        self.radioButton_2.setFont(font1)
        icon1 = QIcon()
        icon1.addFile(u"../../yolov5-7.0/UI\u8bbe\u8ba1/images/\u56fe\u7247\u6536\u96c6_collect-picture.png", QSize(), QIcon.Normal, QIcon.Off)
        self.radioButton_2.setIcon(icon1)
        self.pushButton = QPushButton(Form)
        self.pushButton.setObjectName(u"pushButton")
        self.pushButton.setGeometry(QRect(310, 430, 201, 71))
        self.pushButton.setStyleSheet(u"QPushButton{\n"
"background-color: qlineargradient(spread:pad, x1:0.0227727, y1:0.472, x2:0.977273, y2:0.476955, stop:0 rgba(23, 98, 255, 255), stop:1 rgba(255, 150, 207, 255));\n"
"	font: 75 16pt \"\u5fae\u8f6f\u96c5\u9ed1\";\n"
"color: rgb(255, 255, 255);\n"
"border-radius: 20\n"
"}\n"
"QPushButton:hover{\n"
"background-color:qlineargradient(spread:pad, x1:0.198909, y1:0.302, x2:0.892046, y2:0.687, stop:0 rgba(23, 98, 255, 255), stop:1 rgba(255, 150, 207, 255));\n"
"padding-bottom: 7;\n"
"\n"
"}\n"
"QPushButton:pressed{\n"
"padding-top: 7;\n"
"padding-left: 7;\n"
"}\n"
"QPushButton:disabled{\n"
"background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 rgba(119, 119, 119, 255), stop:1 rgba(221, 221, 221, 255));\n"
"	color: rgba(241, 241, 241, 150);\n"
"}")
        self.comboBox = QComboBox(Form)
        self.comboBox.setObjectName(u"comboBox")
        self.comboBox.setGeometry(QRect(510, 250, 67, 22))
        self.pushButton_2 = QPushButton(Form)
        self.pushButton_2.setObjectName(u"pushButton_2")
        self.pushButton_2.setGeometry(QRect(50, 440, 201, 71))
        self.pushButton_2.setStyleSheet(u"QPushButton{\n"
"background-color: qlineargradient(spread:pad, x1:0.0227727, y1:0.472, x2:0.977273, y2:0.476955, stop:0 rgba(23, 98, 255, 255), stop:1 rgba(255, 150, 207, 255));\n"
"	font: 75 16pt \"\u5fae\u8f6f\u96c5\u9ed1\";\n"
"color: rgb(255, 255, 255);\n"
"border-radius: 20\n"
"}\n"
"QPushButton:hover{\n"
"background-color:qlineargradient(spread:pad, x1:0.198909, y1:0.302, x2:0.892046, y2:0.687, stop:0 rgba(23, 98, 255, 255), stop:1 rgba(255, 150, 207, 255));\n"
"padding-bottom: 7;\n"
"\n"
"}\n"
"QPushButton:pressed{\n"
"padding-top: 7;\n"
"padding-left: 7;\n"
"}\n"
"QPushButton:disabled{\n"
"background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 rgba(119, 119, 119, 255), stop:1 rgba(221, 221, 221, 255));\n"
"	color: rgba(241, 241, 241, 150);\n"
"}")
        self.label_3 = QLabel(Form)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setGeometry(QRect(0, 0, 650, 700))
        self.label_3.setStyleSheet(u"border-radius:30;\n"
"background-color: rgba(255, 255, 255, 150);\n"
"border-image: url(:/images/images/bg.png);\n"
"")
        self.label_3.setScaledContents(True)
        self.pushButton_4 = QPushButton(Form)
        self.pushButton_4.setObjectName(u"pushButton_4")
        self.pushButton_4.setGeometry(QRect(20, 120, 101, 71))
        self.pushButton_4.setStyleSheet(u"QPushButton:hover{\n"
"padding-bottom: 7;\n"
"background-color: rgba(255, 255, 255, 0);\n"
"}\n"
"QPushButton{\n"
"background-color: rgba(255, 255, 255, 0);\n"
"	color: rgb(0, 170, 255);\n"
"	font: 12pt \"\u5fae\u8f6f\u96c5\u9ed1\";\n"
"}\n"
"")
        icon2 = QIcon()
        icon2.addFile(u"images/\u5e2e\u52a9_help.png", QSize(), QIcon.Normal, QIcon.Off)
        self.pushButton_4.setIcon(icon2)
        self.pushButton_4.setIconSize(QSize(40, 40))
        self.pushButton_5 = QPushButton(Form)
        self.pushButton_5.setObjectName(u"pushButton_5")
        self.pushButton_5.setGeometry(QRect(-10, 10, 101, 71))
        self.pushButton_5.setStyleSheet(u"QPushButton:hover{\n"
"padding-bottom: 7;\n"
"background-color: rgba(255, 255, 255, 0);\n"
"}\n"
"QPushButton{\n"
"background-color: rgba(255, 255, 255, 0);\n"
"	color: rgb(0, 170, 255);\n"
"	font: 12pt \"\u5fae\u8f6f\u96c5\u9ed1\";\n"
"}\n"
"")
        icon3 = QIcon()
        icon3.addFile(u"images/logo.png", QSize(), QIcon.Normal, QIcon.Off)
        self.pushButton_5.setIcon(icon3)
        self.pushButton_5.setIconSize(QSize(60, 60))
        self.pushButton_3 = QPushButton(Form)
        self.pushButton_3.setObjectName(u"pushButton_3")
        self.pushButton_3.setGeometry(QRect(530, 20, 101, 71))
        self.pushButton_3.setStyleSheet(u"QPushButton:hover{\n"
"padding-bottom: 7;\n"
"padding-left: 7;\n"
"background-color: rgba(255, 255, 255, 0);\n"
"}\n"
"QPushButton{\n"
"background-color: rgba(255, 255, 255, 0);\n"
"}\n"
"")
        icon4 = QIcon()
        icon4.addFile(u"images/\u5173\u95ed_close.png", QSize(), QIcon.Normal, QIcon.Off)
        self.pushButton_3.setIcon(icon4)
        self.pushButton_3.setIconSize(QSize(40, 40))
        self.label_3.raise_()
        self.radioButton.raise_()
        self.radioButton_2.raise_()
        self.pushButton.raise_()
        self.comboBox.raise_()
        self.pushButton_2.raise_()
        self.pushButton_4.raise_()
        self.pushButton_5.raise_()
        self.pushButton_3.raise_()

        self.retranslateUi(Form)

        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"Login", None))
        self.radioButton.setText(QCoreApplication.translate("Form", u"\u6444\u50cf\u5934\u8bc6\u522b", None))
        self.radioButton_2.setText(QCoreApplication.translate("Form", u"\u56fe\u7247\u8bc6\u522b", None))
        self.pushButton.setText(QCoreApplication.translate("Form", u"\u8c03\u7528detect", None))
        self.pushButton_2.setText(QCoreApplication.translate("Form", u"\u5f00\u59cb\u8bc6\u522b", None))
        self.label_3.setText("")
        self.pushButton_4.setText(QCoreApplication.translate("Form", u"\u5e2e\u52a9", None))
        self.pushButton_5.setText("")
        self.pushButton_3.setText("")
    # retranslateUi