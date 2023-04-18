import os
import shutil
import oyaml as yaml
import pickle
import cv2
import numpy as np

"""
注：当作函数使用
功能：
1.从最新的一次已完成目标检测视频中得到每一帧图片， 并将图片放入训练集
2.更新标签及训练配置（包括训练集和测试集的划分，yaml文件增加类）
参数：name 传入本次更新的类名; new 是否从头开始一个新的配置（会开辟一个新的训练文件夹）
"""


class update:
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

    @staticmethod
    def create_trainFolder(new):
        """
        功能: 创建新的训练文件夹 face{i}
        """

        def check_fileName(FolderPath=f"../train_data"):
            fileNames = os.listdir(FolderPath)
            fileNames.sort(key=lambda x: os.path.getmtime((FolderPath + "\\" + x)))
            if fileNames:
                last = int(fileNames[-1][4:])
                print(last)
                if new:
                    return f'face{last + 1}'
                else:
                    return f'face{last}'
            else:
                return 'face'

        FolderPath = f"../train_data/{check_fileName()}"
        if new:
            list(os.makedirs(f"{FolderPath}/{i}/{j}", exist_ok=True) for i in ['images', 'labels'] for j in
                 ['train', 'test'])

        return FolderPath

    def __init__(self, name, new=False):
        """
        如果new为True，则新开辟一个训练文件夹，否则默认为上一个开辟的文件夹
        yaml中的字典通过class_dict.txt初始化和读取
        """
        self.filePath = self.get_path("../runs/detect")
        self.trainFolder_Path = self.create_trainFolder(new)
        if new:
            self.data = {"path": f"{self.trainFolder_Path[1: ]}", "train": 'images/train', "val": "images/train",
                         'test': '',
                         "names": {}}

            with open(f'{self.trainFolder_Path}/class_dict.txt', 'wb') as f:
                f.write(pickle.dumps(self.data))

            self.cls = 0
            self.update(name)
        else:
            with open(f'{self.trainFolder_Path}/class_dict.txt', 'rb') as f:
                self.data = pickle.loads(f.read())
                print(self.data)
                keys = list(self.data['names'].keys())
                values = list(self.data['names'].values())
                if not (name in values):
                    if keys:
                        self.cls = keys[-1] + 1

                    self.update(name)

    def update(self, name: str):
        """
        更新yaml，label class和images
        """

        def update_yaml():
            """
            更新yaml，并将更新后的数据重新写入class_dict.txt
            """
            self.data["names"].setdefault(self.cls, name)
            print(self.data['names'])
            with open(f'{self.trainFolder_Path}/class_dict.txt', 'wb') as f:
                f.write(pickle.dumps(self.data))
            with open("../data/est.yaml", "w") as f:
                yaml.safe_dump(self.data, f)

        def change_label_class():
            """
            更改类名，如1：'LCF', 2: 'MMMMMCREE'
            """

            def label_write(labelsName, choice):
                choice = (lambda i: 'train' if i == 0 else 'test')(choice)
                for labelName in labelsName:
                    path = labelsPath + '/' + labelName
                    with open(path, 'r') as f:
                        text = f.read()
                    b_text = bytearray(text, 'utf-8')
                    b_text.pop(0)
                    list(b_text.insert(0, int(i) + 48) for i in reversed(str(self.cls)))
                    text = b_text.decode('utf-8')
                    with open(f'{self.trainFolder_Path}/labels/{choice}/{self.cls}' + labelName[1:], 'w') as f:
                        f.write(text)

            labelsPath = self.filePath + '/labels'
            labelsName = os.listdir(labelsPath)
            split_list = np.split(labelsName, indices_or_sections=[int(0.8 * len(labelsName))])
            list(label_write(li, choice) for choice, li in enumerate(split_list))

        def video_split():
            """
            逐帧截取视频内容，并写入训练集
            """
            cap = cv2.VideoCapture(f'{self.filePath}' + '/0.mp4')
            i = 0
            while (cap.isOpened()):
                ret, frame = cap.read()
                if ret == True:
                    print(f"\r{self.cls}_{i}.jpg: Done!", end="")
                    cv2.imwrite(f'{self.trainFolder_Path}/images/train/{self.cls}_' + str(i) + '.jpg', frame)
                else:
                    break
                i += 1
            cap.release()
            # 按照标签测试集的内容，将图片训练集中的一部分转移到图片测试集中
            for name in os.listdir(f'{self.trainFolder_Path}/labels/test'):
                if name[0: len(str(self.cls))] == str(self.cls):
                    jpg_name = name.split('.')[0] + '.jpg'
                    shutil.move(f'{self.trainFolder_Path}/images/train/{jpg_name}',
                                f'{self.trainFolder_Path}/images/test/{jpg_name}')

        # 根据转移逻辑，标签操作应在图片操作之前
        change_label_class()
        video_split()
        update_yaml()


if __name__ == "__main__":
    update('2', new=True)
