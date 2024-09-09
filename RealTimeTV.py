import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

gestures = {"videodata\\Close":"fist",
            "videodata\\Thumbs_Up":"thumbs up",
            "videodata\\Thumbs_Down":"thumbs down",
            "videodata\\Peace":"peace",
            "videodata\\Rock":"rock",
            "videodata\\OK":"okay",
            "videodata\\Call_Me":"call me",
            "videodata\\Smile":"smile",
            "videodata\\Stop":"stop",
            "videodata\\Live_Long":"live long"}


rezultati = {}
correct_all = 0
wrong_all = 0
one_frame = 0

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

model = load_model('mp_hand_gesture')

# Load class names
with open("gesture.names", "r") as f:
    classNames = f.read().split('\n')

for key, value in gestures.items():
    image_files = [f for f in os.listdir(key) if os.path.isfile(os.path.join(key, f))]
    correct = 0
    wrong = 0
    for image_file in image_files:
        image_path = os.path.join(key, image_file)
        frame = cv2.imread(image_path)
        x, y, c = frame.shape
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = hands.process(framergb)
        className = ''

        if result.multi_hand_landmarks:
            landmarks = []
            for handslms in result.multi_hand_landmarks:
                for lm in handslms.landmark:
                    lmx = int(lm.x * x)
                    lmy = int(lm.y * y)
                    landmarks.append([lmx, lmy])

                # mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

                prediction = model.predict([landmarks])
                classID = np.argmax(prediction)
                className = classNames[classID]
                one_frame = 0
        else:
            one_frame += 1

        # cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
        #            1, (0, 0, 255), 2, cv2.LINE_AA)
        # cv2.imshow("Output", frame)
        # if cv2.waitKey(0) == ord('q'):
        #     continue

        if className == value and one_frame <= 1:
            correct += 1
            correct_all += 1
        elif one_frame > 1:
            continue
        else:
            wrong += 1
            wrong_all += 1

    rezultati.update({value: (correct, wrong)})

postotak_c = round((correct_all / (correct_all + wrong_all)) * 100, 2)
postotak_w = round((wrong_all / (correct_all + wrong_all)) * 100, 2)
print(rezultati)
print(f"Correct: {correct_all} ({postotak_c}%)\nWrong: {wrong_all} ({postotak_w}%)")
cv2.destroyAllWindows()
