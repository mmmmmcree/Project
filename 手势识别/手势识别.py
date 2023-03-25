import cv2
import mediapipe as mp


cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands(min_tracking_confidence=0.3, min_detection_confidence=0.5)
mpDraw = mp.solutions.drawing_utils
markStyle = mpDraw.DrawingSpec(color=(0, 0, 255), thickness=4)
lineStyle = mpDraw.DrawingSpec(color=(255, 255, 255), thickness=2)


while True:
    ret, img = cap.read()
    if ret:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)
        multi_hand_landmarks = result.multi_hand_landmarks
        if multi_hand_landmarks:
            for hand_landmarks in multi_hand_landmarks:
                print(len(hand_landmarks.landmark))

        cv2.imshow("img", img)
    if cv2.waitKey(1) == ord('q'):
        break