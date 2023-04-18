from PySide2 import QtCore
from PySide2.QtCore import QObject, Signal
from PySide2.QtWidgets import QApplication, QMessageBox, QPlainTextEdit, QTextBrowser, QLabel, QFileDialog, QMainWindow, \
    QInputDialog, QWidget, QLineEdit
from PySide2.QtUiTools import QUiLoader
from PySide2.QtGui import QImage, QPixmap, QIcon, Qt
import time
import cv2
import numpy as np
from paddleocr import PaddleOCR
from PIL import Image
import os
from tqdm.tk import tqdm
from rich import print
from threading import Thread, Timer
import multiprocessing
from LIB.share import Shared_Info
from detect副本 import main, parse_opt
from uptate import update
from os import startfile
from datetime import datetime
from collections import Counter


class RepeatingTimer(Timer):
    """
    间隔一段时间执行某个函数的线程
    """

    def run(self):
        while not self.finished.is_set():
            self.function(*self.args, **self.kwargs)
            self.finished.wait(self.interval)


class MySignals(QObject):
    """
    信号类
    多线程时某个控件发送信号，由信号接收器接收该信号并执行某函数。
    """
    text_print = Signal(QTextBrowser, str)  # 打印文字信号
    photo_print = Signal(QLabel, np.ndarray)  # 打印图片信号


class Win_Login:
    """
    第一版实现的Login，即基础功能演示的Login界面
    """

    def __init__(self):
        _, self.parser = parse_opt()  # 加载detect的参数表
        self.ui = QUiLoader().load('Login.ui')
        self.ui.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        self.ui.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.choice = None  # 摄像头识别或图片识别的选项
        self.p = None  # 用于detect的进程
        self.ui.pushButton_2.setEnabled(False)
        # 以下是各控件及其连接的函数
        self.ui.pushButton.setEnabled(False)
        self.ui.pushButton_2.clicked.connect(self.main_start)
        self.ui.pushButton.clicked.connect(self.detect)
        self.ui.pushButton_3.clicked.connect(self.quit)
        self.ui.pushButton_4.clicked.connect(self.help_message)
        self.ui.pushButton_5.clicked.connect(self.info_message)
        self.ui.comboBox.currentIndexChanged.connect(self.handleSelectionChange)
        self.ui.buttonGroup.buttonClicked.connect(self.handleButtonClicked)

    def handleSelectionChange(self):
        """模型选择"""
        weights = self.ui.comboBox.currentText()
        self.parser.set_defaults(weights=f'weights/{weights}.pt')

    def quit(self):
        """返回"""
        if self.p:  # 如果有detect进程则关闭该进程
            self.p.terminate()
        # 切换界面至Win_Choice
        Shared_Info.Login = Win_Choice()
        self.ui.close()
        Shared_Info.Login.ui.show()

    @staticmethod
    def info_message():
        with open('texts/info.txt', mode='r') as f:
            text = f.read()
        QMessageBox.about(QMainWindow(), '出品信息', text)

    @staticmethod
    def help_message():
        with open('texts/help.txt', mode='r') as f:
            text = f.read()
        QMessageBox.about(QMainWindow(), '使用说明', text)

    def handleButtonClicked(self):
        """对应界面中两个小圆点选项的选择"""
        chosen_button = self.ui.buttonGroup.checkedButton()
        choice = chosen_button.text()
        if choice == '摄像头识别':
            self.choice = '摄像头识别'
            self.ui.pushButton.setEnabled(True)
        elif choice == '图片识别':
            self.choice = '图片识别'
            self.ui.pushButton.setEnabled(True)

    def detect(self):
        """调用detect"""
        if self.choice == '图片识别':
            self.parser.set_defaults(save_txt=True)
            self.parser.set_defaults(source='../data/images')

            choice = QMessageBox.question(  # 弹出对话框，询问是否选择展示检测完的图片
                self.ui,
                '确认',
                '是否展示检测图片?')

            if choice == QMessageBox.Yes:
                self.parser.set_defaults(view_img=True)

            self.ui.pushButton.setEnabled(False)
            self.ui.radioButton.setEnabled(False)
            self.ui.radioButton_2.setEnabled(False)

            opt = self.parser.parse_args()
            main(opt)

            self.ui.pushButton_2.setEnabled(True)
            QMessageBox.about(QMainWindow(), '提示', '目标检测完毕，可以开始识别了！')

        elif self.choice == '摄像头识别':
            self.parser.set_defaults(save_txt=True)
            self.parser.set_defaults(save_crop=True)
            self.parser.set_defaults(source='0')

            QMessageBox.about(QMainWindow(), '提示', '实时目标检测成功调用，等待识别按钮亮起后开始识别。按q退出实时目标检测。')
            self.ui.pushButton.setEnabled(False)
            self.ui.radioButton.setEnabled(False)
            self.ui.radioButton_2.setEnabled(False)

            opt = self.parser.parse_args()
            self.ui.pushButton_2.setEnabled(True)
            self.p = multiprocessing.Process(target=main, args=(opt,))
            self.p.start()

    def main_start(self):
        """切换界面至Win_Main"""
        Shared_Info.MainWindow = Win_Main(self.choice)
        Shared_Info.MainWindow.ui.show()
        self.ui.close()


class Win_Main:
    """基础功能演示文字识别主界面"""

    def __init__(self, choice):
        self.ui = QUiLoader().load('MainWindow.ui')
        self.ocr = PaddleOCR(use_angle_cls=True, lang="ch", use_gpu=True)  # 调用PaddleOCR的识别模型
        self.ui.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        self.ui.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.choice = choice  # 摄像头识别 or 图片识别
        self.autoCapture = None  # 是否开启自动抓拍，对应RepeatingTimer类
        self.ms = MySignals()  # 调用信号类
        self.path = self.get_path("../runs/detect")  # 得到图片存储的路径

        if self.choice == '摄像头识别':
            self.task = self.task2  # self.task与按钮连接，根据不同的选择连接不同的任务（task1 or task2）
            """
            加入多功能选项时， 将创建列表归于功能下
            """
            self.detected_plates = dict()

            self.autoCapture = QMessageBox.question(self.ui, '确认', '是否需要定时抓拍？')
            if self.autoCapture == QMessageBox.Yes:  # 如果选择自动抓拍，则创建一个RepeatingTimer线程
                self.t = RepeatingTimer
                self.t = RepeatingTimer(1.5, self.task)
                self.t.start()
        elif self.choice == '图片识别':
            self.task = self.task1
        # 以下为按钮连接
        self.ui.pushButton.clicked.connect(self.task)
        self.ui.pushButton_2.clicked.connect(self.backToLogin)
        # 以下为信号连接
        self.ms.text_print.connect(self.printTextToGui)
        self.ms.photo_print.connect(self.printPhotoToGui)

    def backToLogin(self):
        """返回Win_Login界面"""
        Shared_Info.Login = Win_Login()
        self.ui.close()
        Shared_Info.Login.ui.show()
        if self.autoCapture:  # 如果选择自动抓拍，在返回前关闭线程
            self.t.cancel()

    @staticmethod
    def printTextToGui(widget, text):
        """将文字输入到控件中"""
        widget.append(str(text))
        widget.ensureCursorVisible()

    @staticmethod
    def printPhotoToGui(widget, img):
        """将图片输入到控件中"""
        image = QImage(img, img.shape[1], img.shape[0], QImage.Format_BGR888)
        image = QPixmap(image).scaled(400, 300)
        widget.setPixmap(image)
        widget.setScaledContents(True)

    def task2(self):
        """摄像头识别所对应的task"""

        def txt_recognize(path, ocr):
            """文字识别"""
            try:
                labels_path, label_lists = self.sortList_byTime(path, 'labels')  # 将得到的标签按照时间顺序升序排序
                plate_path, img_lists = self.sortList_byTime(path, 'crops\\Plate')  # 将得到的图片按照时间顺序升序排序
                txtName = label_lists[-1]  # 得到最新获取的图片的标签文件名
                with open(os.path.join(labels_path, txtName)) as f:
                    texts = f.readlines()
                    num = len(texts)  # 通过标签中的数据条数确定图片中车牌的个数
                img_names = img_lists[-num:]  # 图片列表中最后的num个图片即为该单个图片中车牌的截图
                for imgName in img_names:  # 对每个截图进行ocr识别，将结果打印到控件上
                    img_path = os.path.join(plate_path, imgName)
                    img = cv2.imread(img_path)
                    result = ocr.ocr(img, cls=True)
                    try:
                        print(f'[green]{result[0][0][1][0]}[/green]')
                        self.ms.text_print.emit(self.ui.textBrowser, result[0][0][1][0])
                    except Exception:
                        print('[red]识别失败[/red]')
                        self.ms.text_print.emit(self.ui.textBrowser, '识别失败')
                list(map(lambda x: os.remove(os.path.join(plate_path, x)), img_lists[0: -1]))  # 识别完毕后删除所有该识别图片之前的图片
                list(map(lambda x: os.remove(os.path.join(labels_path, x)), label_lists[0: -1]))  # 删除所有该识别图片对应标签之前的标签
            except:
                self.ms.text_print.emit(self.ui.textBrowser, '尚未识别到车牌')

        # 以新建一个线程的方式执行该task
        thread = Thread(target=txt_recognize, args=(self.path, self.ocr))
        thread.start()

    def task1(self):
        """图片识别所对应的task"""

        def get_plate_imgs(path):
            """
            通过标签中的数据得到对应截图（这是一个生成器，yield车牌截图的个数以及各个车牌截图）
            当时用YOLOv5_v5.0写的，没有直接获取车牌截图的参数
            """
            labels_path, label_lists = self.sortList_byTime(path, 'labels')
            for txtName in label_lists:
                with open(os.path.join(labels_path, txtName)) as f:
                    texts = f.readlines()
                    yield len(texts)  # yield车牌截图的个数
                    imgName = txtName.split('.')[0] + '.jpg'
                    img_path = os.path.join(path, imgName)
                    assert os.path.exists(img_path), "image file {} dose not exist.".format(img_path)
                    Img = cv2.imread(img_path)

                    self.ms.photo_print.emit(self.ui.label_2,
                                             np.asarray(Image.fromarray(Img).resize((460, 460), Image.CUBIC)))

                    for text in texts:
                        text = text.replace('\n', ' ').split(' ')
                        x_c, y_c, w, h = float(text[1]), float(text[2]), float(text[3]), float(text[4])
                        w, h, x_c, y_c = w * Img.shape[1], h * Img.shape[0], x_c * Img.shape[1], y_c * Img.shape[0]
                        xmin, xmax, ymin, ymax = x_c - w / 2, x_c + w / 2, y_c - h / 2, y_c + h / 2
                        img = Image.fromarray(Img)
                        img = img.crop((xmin, ymin, xmax, ymax))
                        img = img.resize((100, 30), Image.LANCZOS)
                        img = np.asarray(img)
                        yield img  # yield车牌截图

        def txt_recognize(path, ocr):
            """文字识别"""
            with open('识别结果.txt', mode='w') as f:  # 将识别结果写入"识别结果.txt"中
                imgs_info = get_plate_imgs(path)  # 激活生成器
                rec_success, rec_failed, plates_total = 0, 0, 0
                dir_list = os.listdir(path)[0:]
                dir_list.remove('labels')
                dir_list.sort(key=lambda x: os.path.getmtime((path + "\\" + x)))
                img_total = len(dir_list)
                for imageName in tqdm(dir_list, desc='正在识别车牌', leave=False, mininterval=0.0001):
                    plate_num = next(imgs_info)  # 得到截图个数
                    plates_total += plate_num
                    for i in range(plate_num):
                        img = next(imgs_info)  # 得到对应的各个图片
                        self.ms.photo_print.emit(self.ui.label, img)
                        result = ocr.ocr(img, cls=True)
                        try:
                            s = f'{imageName}中第{i + 1}个车牌:{result[0][0][1][0].replace(".", "·")}'
                            self.ms.text_print.emit(self.ui.textBrowser, s)
                            f.write(s + '\n')
                            rec_success += 1
                        except Exception:
                            s = f'{imageName}中第{i + 1}个车牌 识别失败'
                            self.ms.text_print.emit(self.ui.textBrowser, s)
                            f.write(s + '\n')
                            rec_failed += 1
            startfile('识别结果.txt')  # 打开"识别结果.txt"
            return rec_success, rec_failed, img_total, plates_total

        thread = Thread(target=txt_recognize, args=(self.path, self.ocr))
        thread.start()

    @staticmethod
    def get_path(dir: str):
        fileNames = os.listdir(dir)
        fileNames.sort(key=lambda x: os.path.getmtime((dir + "\\" + x)))
        fileName = fileNames[-1]
        path = os.path.join(dir, fileName)
        return path

    @staticmethod
    def sortList_byTime(path, dirName: str):
        """按时间顺序升序排序"""
        new_path = os.path.join(path, dirName)
        lists = os.listdir(new_path)
        lists.sort(key=lambda x: os.path.getmtime((new_path + "\\" + x)))
        return new_path, lists


class Win_Choice:
    """初始选择界面"""
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
        """跳转至选择对应界面"""
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


class Parking_Login:
    """停车场管理登录界面"""
    def __init__(self):
        _, self.parser = parse_opt()  # 调用detect的参数表
        self.parser.set_defaults(weights='weights/best_gray.pt')  # 设置权重参数为best_gray.pt
        self.ui = QUiLoader().load('ParkingLogin.ui')
        self.ui.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        self.ui.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.ui.pushButton.setEnabled(True)
        self.ui.pushButton_2.setEnabled(False)
        self.p = None  # 为detect准备的进程
        self.ui.pushButton_2.clicked.connect(self.main_start)
        self.ui.pushButton.clicked.connect(self.detect)
        self.ui.pushButton_3.clicked.connect(self.quit)
        self.ui.pushButton_4.clicked.connect(self.help_message)
        self.ui.pushButton_5.clicked.connect(self.info_message)

    def quit(self):
        if self.p:
            self.p.terminate()
        Shared_Info.Login = Win_Choice()
        self.ui.close()
        Shared_Info.Login.ui.show()

    @staticmethod
    def info_message():
        with open('texts/info.txt', mode='r') as f:
            text = f.read()
        QMessageBox.about(QMainWindow(), '出品信息', text)

    @staticmethod
    def help_message():
        with open('texts/help_parkinglot.txt', mode='r') as f:
            text = f.read()
        QMessageBox.about(QMainWindow(), '使用说明', text)

    def detect(self):
        """调用detect"""
        self.parser.set_defaults(save_txt=True)  # 保存标签
        self.parser.set_defaults(save_crop=True)  # 保存截图
        self.parser.set_defaults(source='0')  # 识别设备为摄像头

        QMessageBox.about(QMainWindow(), '提示', '实时目标检测成功调用，等待识别按钮亮起后开始识别。按q退出实时目标检测。')

        opt = self.parser.parse_args()

        self.p = multiprocessing.Process(target=main, args=(opt,))  # detect进程
        self.p.start()

        self.ui.pushButton.setEnabled(False)
        self.ui.pushButton_2.setEnabled(True)

    def main_start(self):
        Shared_Info.MainWindow = Parking_lot()
        Shared_Info.MainWindow.ui.show()
        self.ui.close()


class Parking_lot:
    """停车场管理主界面"""
    def __init__(self):
        self.ui = QUiLoader().load('ParkingMain.ui')
        self.ocr = PaddleOCR(use_angle_cls=True, lang="ch", use_gpu=True)
        self.ui.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        self.ui.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.ms = MySignals()
        self.path = self.get_path("../runs/detect")

        self.detected_plates = dict()  # 进入停车场车辆的容器，保存对应车牌号和进入时间

        self.ui.pushButton.clicked.connect(self.task)
        self.ui.pushButton_2.clicked.connect(self.backToLogin)

        self.ui.spinBox.setValue(2)  # 当前空余车位的初始值设为2

        self.ui.spinBox.valueChanged.connect(self.valueChange)

        self.ms.text_print.connect(self.printTextToGui)
        self.ms.text_print.emit(self.ui.textBrowser_2, f'当前车位空余: {self.ui.spinBox.value()}个')

    def valueChange(self):
        """剩余车位的值改变后对应提示"""
        self.ms.text_print.emit(self.ui.textBrowser_2, f'当前车位空余: {self.ui.spinBox.value()}个')
        if self.ui.spinBox.value() <= 0:
            self.ui.spinBox.setValue(0)
            self.ms.text_print.emit(self.ui.textBrowser_2, '车位已满，禁止进入')

    @staticmethod
    def sortList_byTime(path, dirName: str):
        new_path = os.path.join(path, dirName)
        lists = os.listdir(new_path)
        lists.sort(key=lambda x: os.path.getmtime((new_path + "\\" + x)))
        return new_path, lists

    @staticmethod
    def get_path(dir: str):
        fileNames = os.listdir(dir)
        fileNames.sort(key=lambda x: os.path.getmtime((dir + "\\" + x)))
        fileName = fileNames[-1]
        path = os.path.join(dir, fileName)
        return path

    def backToLogin(self):
        Shared_Info.Login = Parking_Login()
        self.ui.close()
        Shared_Info.Login.ui.show()

    @staticmethod
    def printTextToGui(widget, text):
        widget.append(str(text))
        widget.ensureCursorVisible()

    def task(self):

        def plate_analyze(rec_result):
            """车辆进入或离开对应提示"""
            if not (rec_result in self.detected_plates) and self.ui.spinBox.value() != 0:  # 如果是新车辆进入(对应第一次检测到)
                self.detected_plates[rec_result] = time.time()  # 记录进入时间
                text = f'车牌号为{rec_result}的车辆于{datetime.now()}进入'
                self.ms.text_print.emit(self.ui.textBrowser, text)
                self.ui.spinBox.setValue(self.ui.spinBox.value() - 1)  # 剩余车位 - 1

            elif rec_result in self.detected_plates:  # 如果是车辆离开(对应第二次检测到)
                time_elapsed = int(time.time() - self.detected_plates[rec_result])  # 计算停放时间
                price = time_elapsed * 100
                text = f'车牌号为{rec_result}的车辆于{datetime.now()}离开, 共计{time_elapsed}秒, 停车费为{price}元'
                self.ms.text_print.emit(self.ui.textBrowser, text)
                del self.detected_plates[rec_result]
                self.ui.spinBox.setValue(self.ui.spinBox.value() + 1)  # 剩余车位 + 1

            elif self.ui.spinBox.value() == 0:  # 如果剩余车位为0，则禁止进入
                self.ms.text_print.emit(self.ui.textBrowser_2, '车位已满，禁止进入')

        def get_plate_imgs_crop(path, ocr):
            """获得车牌图片"""
            rec_results = list()
            for i in range(5):
                """
                对于同一辆车截取五张图片进行识别
                原本想使用这种方法防止只使用一张图片识别容易出错，导致字典中车牌号记录出错而无法判断车辆是否离开的问题
                然而好像没啥用
                """
                time.sleep(0.02)
                try:
                    labels_path, label_lists = self.sortList_byTime(path, 'labels')
                    plate_path, img_lists = self.sortList_byTime(path, 'crops\\Plate')
                    txtName = label_lists[-1]
                    with open(os.path.join(labels_path, txtName)) as f:
                        texts = f.readlines()
                        num = len(texts)
                    img_names = img_lists[-num:]
                    for imgName in img_names:
                        img_path = os.path.join(plate_path, imgName)
                        img = cv2.imread(img_path)
                        result = ocr.ocr(img, cls=True)
                        rec_result = result[0][0][1][0]
                        rec_results.append(rec_result)
                except:
                    pass
            if not rec_results:
                self.ms.text_print.emit(self.ui.textBrowser, '尚未识别到车牌')
            else:
                rec_result = Counter(rec_results).most_common(1)[0][0]
                plate_analyze(rec_result)
                list(map(lambda x: os.remove(os.path.join(plate_path, x)), img_lists))
                list(map(lambda x: os.remove(os.path.join(labels_path, x)), label_lists))

        thread = Thread(target=get_plate_imgs_crop, args=(self.path, self.ocr))
        thread.start()


class Face_Login:
    """人脸识别登录界面"""
    def __init__(self):
        _, self.parser = parse_opt()

        self.ui = QUiLoader().load('FaceLogin.ui')
        self.ui.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        self.ui.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.ui.pushButton.setEnabled(True)
        self.cls_name = None  # 用于方法update()
        self.ui.pushButton.clicked.connect(self.get_face_data)
        self.ui.pushButton_2.clicked.connect(self.face_rec)
        self.ui.pushButton_3.clicked.connect(self.quit)

    def quit(self):
        Shared_Info.Login = Win_Choice()
        self.ui.close()
        Shared_Info.Login.ui.show()

    def face_rec(self):
        self.parser.set_defaults(source='0')
        weight_path = self.get_path('../runs/train/') + '/weights/best.pt'
        self.parser.set_defaults(weights=weight_path)
        opt = self.parser.parse_args()
        main(opt)

    def get_face_data(self):
        """人脸数据录入"""
        self.parser.set_defaults(save_txt=True)
        self.parser.set_defaults(save_crop=True)
        self.parser.set_defaults(source='0')
        self.parser.set_defaults(weights='weights/face.pt')  # face.pt为专门检测人脸的模型参数

        self.ui.pushButton.setEnabled(False)
        self.ui.pushButton_2.setEnabled(False)
        QMessageBox.about(QMainWindow(), '提示', '务必按q结束检测，否则无法完成录入。\n接下来请输入分类标签，用英文人名标记本次输入对象')
        self.cls_name, okPressed = QInputDialog.getText(self.ui, '分类标签', '输入分类标签:', text='face')  # 用于获得本次录入的分类标签和类名
        while (not okPressed) or (self.cls_name == 'face'):
            self.cls_name, okPressed = QInputDialog.getText(self.ui, '分类标签', '输入分类标签:', text='face')
        new = QMessageBox.question(self.ui, '确认', '是否开始一个新的训练文件夹配置？')  # 是否开辟一个新的训练文件夹
        new = (lambda choice: True if choice == QMessageBox.StandardButton.Yes else False)(new)

        opt = self.parser.parse_args()
        main(opt)
        print('detect over')
        update(self.cls_name, new)  # 方法update()详见update.py
        self.ui.pushButton.setEnabled(True)
        self.ui.pushButton_2.setEnabled(True)

    @staticmethod
    def get_path(dir: str):
        """
        功能: 按照时间顺序得到最新的文件夹路径
        """
        fileNames = os.listdir(dir)
        fileNames.sort(key=lambda x: os.path.getmtime((dir + "\\" + x)))
        fileName = fileNames[-1]
        path = os.path.join(dir, fileName)
        return path


class Face_Rec:
    """还没实现"""
    def __init__(self):
        self.ui = QUiLoader().load('FaceMain.ui')
        self.ocr = PaddleOCR(use_angle_cls=True, lang="ch", use_gpu=True)
        self.ui.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        self.ui.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.ms = MySignals()


def LOGIN_UI_SHOW():
    """打开初始界面"""
    app = QApplication([])
    app.setWindowIcon(QIcon('images/logo.png'))
    Shared_Info.Login = Win_Choice()
    Shared_Info.Login.ui.show()
    app.exec_()


if __name__ == "__main__":
    LOGIN_UI_SHOW()
