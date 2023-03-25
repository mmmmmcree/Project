import cv2

cap = cv2.VideoCapture('D:/YOLOv5_7.0/runs/detect/exp017/0.mp4')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = cap.get(cv2.CAP_PROP_FPS)
# fps=30
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
# size=(960,544)
i = 0
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        print(1)
        cv2.imwrite('D:/YOLOv5_7.0/train_data/face1/images/train/0_' + str(i) + '.jpg', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
    i = i + 1
cap.release()
