import cv2
import mediapipe as mp
import pandas as pd
import torch

from My_nn import My_nn

device = torch.device("cuda")
model = torch.load("my_nn.pth").to(device)


class Hand_Detect:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.hands = mp.solutions.hands.Hands(min_tracking_confidence=0.3, min_detection_confidence=0.6)  # 模型
        self.mpDraw = mp.solutions.drawing_utils  # 与模型配套的画图工具
        self.img_width = None
        self.img_height = None
        self.dataframe = None


    def get_result(self, img):
        """传入图片, 得到解析后的结果"""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.hands.process(img_rgb)  # 对img转rgb后的图片解析
        return result

    def get_pos(self, landmarks):
        """得到一只手的坐标集合里的每个坐标的整数(x, y), 用于画图"""
        for i, landmark in enumerate(landmarks.landmark):  # 得到一只手的坐标集合里的每一个坐标
            yield i, landmark.x, landmark.y, landmark.z  # x, y为坐标在图中的比例坐标，z好像是距离手腕的深度什么的，忘了

    def draw(self, multi_hand_landmarks, img):
        """在img上画图，返回画好的img"""

        def add_element(i, x, y):
            """在img上对各结点添加一些特殊元素，如在所有结点上添加文字、用圆标记大拇指等。x, y是图片上的整数坐标点"""
            if i % 4 == 0 and i != 0:
                cv2.circle(img, (x, y), radius=10, color=(0, 0, 220), thickness=cv2.FILLED)  # 为指尖画上圆
            cv2.putText(img, str(i), (x - 20, y - 20), cv2.FONT_HERSHEY_PLAIN, fontScale=0.4,
                        color=(0, 255, 255), thickness=1)  # 给所有节点写上序号

        """以下为draw的主程序"""
        markStyle = self.mpDraw.DrawingSpec(color=(0, 0, 255), thickness=4)  # 标记节点的风格
        lineStyle = self.mpDraw.DrawingSpec(color=(255, 0, 0), thickness=2)  # 节点连线的风格
        for single_hand_landmarks in multi_hand_landmarks:  # 得到每只手的节点坐标
            self.mpDraw.draw_landmarks(img, single_hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS,
                                       markStyle, lineStyle)  # 画出所有坐标对应的结点，并将这些结点间画上线来连接
            gen_pos = self.get_pos(single_hand_landmarks)
            list(add_element(i, int(x * self.img_width), int(y * self.img_height)) for i, x, y, _ in
                 gen_pos)  # 对所有节点进行add_element的操作
        return img

    def get_dataframe(self, multi_hand_landmarks, dataframe, label):
        landmarks_list = list()
        for single_hand_landmarks in multi_hand_landmarks:  # 得到每只手的节点坐标
            gen_pos = self.get_pos(single_hand_landmarks)
            list(landmarks_list.extend([x, y, z]) for _, x, y, z in gen_pos)

        df = pd.DataFrame(landmarks_list, index=None, columns=None).T
        df["label"] = label
        dataframe = pd.concat([dataframe, df], ignore_index=True)
        return dataframe

    def predict(self, multi_hand_landmarks):
        landmarks_list = list()
        for single_hand_landmarks in multi_hand_landmarks:  # 得到每只手的节点坐标
            gen_pos = self.get_pos(single_hand_landmarks)
            list(landmarks_list.extend([x, y, z]) for _, x, y, z in gen_pos)
        input = torch.Tensor(landmarks_list)
        input = input.to(device)
        output = model(input).item()

        print(round(output, 1))

    def detect(self, label=0):
        """调用cap来得到手的图片,并显示经过处理的图片, label参数为是否选择要获得坐标数据的dataframe"""
        if label:
            self.dataframe = pd.DataFrame(index=None, columns=None)

        while True:
            ret, img = self.cap.read()
            if ret:
                self.img_width = img.shape[1]
                self.img_height = img.shape[0]
                result = self.get_result(img)
                multi_hand_landmarks = result.multi_hand_landmarks
                """
                multi_hand_landmarks为解析结果的坐标信息，是一个列表
                其中元素的数量就是一张图片内检测到的手的数量
                每个元素是所有21个结点信息的landmark"集合"
                """
                if multi_hand_landmarks:  # 如果检测到手, 则用draw对手做一些标记
                    img = self.draw(multi_hand_landmarks, img)
                    if label:
                        self.dataframe = self.get_dataframe(multi_hand_landmarks, self.dataframe, label)
                    else:
                        try:
                            self.predict(multi_hand_landmarks)
                        except Exception:
                            pass
                cv2.imshow("img", img)
            if cv2.waitKey(1) == ord('q'):
                if label:
                    print(self.dataframe)
                    self.dataframe.to_csv(f"Hand Landmarks data/label {label}.csv")

                break


if __name__ == "__main__":
    hd = Hand_Detect()
    # hd.detect(label=2)
    hd.detect()
