import cv2
import mediapipe as mp  # MediaPipe==0.10.0 ili novije
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import os
import math

gestures = {"videodata\\Pointing_Up":"Pointing_Up",
            "videodata\\Open":"Open_Palm",
            "videodata\\Close":"Closed_Fist",
            "videodata\\Thumbs_Up":"Thumb_Up",
            "videodata\\Thumbs_Down":"Thumb_Down",
            "videodata\\Peace":"Victory",
            "videodata\\Rock":"ILoveYou"}

global rezultati, correct_all, wrong_all, prev_gesture, one_frame
rezultati = {}
correct_all = 0
wrong_all = 0
prev_gesture = None
one_frame = 0

# Define global variables
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Desired dimensions for displaying images
DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480

def resize_and_show(image):
    h, w = image.shape[:2]
    if h < w:
        img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h / (w / DESIRED_WIDTH))))
    else:
        img = cv2.resize(image, (math.floor(w / (h / DESIRED_HEIGHT)), DESIRED_HEIGHT))
    return img


def start_the_test(images, results, value):
    global rezultati, correct_all, wrong_all, prev_gesture, one_frame
    correct = 0
    wrong = 0
    for i, (image, (gesture, multi_hand_landmarks)) in enumerate(zip(images.values(), results)):
        if gesture is not None:
            title = f"{gesture.category_name} ({gesture.score:.2f})"
        else:
            title = "No Hand Detected"

        annotated_image = image.copy()

        if multi_hand_landmarks:
            for hand_landmarks in multi_hand_landmarks:
                hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                hand_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in
                    hand_landmarks
                ])

        #         mp_drawing.draw_landmarks(
        #             annotated_image,
        #             hand_landmarks_proto,
        #             mp_hands.HAND_CONNECTIONS,
        #             mp_drawing_styles.get_default_hand_landmarks_style(),
        #             mp_drawing_styles.get_default_hand_connections_style()
        #         )
        #
        # cv2.putText(annotated_image, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        #
        # resized_image = resize_and_show(annotated_image)
        # cv2.imshow(f"Image {i + 1}", resized_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        if gesture and gesture.category_name == value:
            correct += 1
            correct_all += 1
            prev_gesture = gesture.category_name
            one_frame = 0
        elif not gesture and prev_gesture == value and one_frame < 1:
            correct += 1
            correct_all += 1
            one_frame += 1
        elif one_frame >= 1:
            continue
        else:
            wrong += 1
            wrong_all += 1
            if gesture:
                one_frame = 0
            else:
                one_frame += 1

    rezultati.update({value: (correct, wrong)})

# Create GestureRecognizer object
base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)

for key, value in gestures.items():
    images = {name: cv2.imread(os.path.join(key, name)) for name in os.listdir(key)}
    images_for_processing = []
    results = []

    for image_file_name in images.keys():
        # Load input image
        image = mp.Image.create_from_file(os.path.join(key, image_file_name))

        # Recognize gestures in the input image
        recognition_result = recognizer.recognize(image)

        # Check if any gestures were detected
        if recognition_result.gestures and recognition_result.gestures[0]:
            # Process result if a gesture is detected
            images_for_processing.append(image)
            top_gesture = recognition_result.gestures[0][0]
            hand_landmarks = recognition_result.hand_landmarks
            results.append((top_gesture, hand_landmarks))
        else:
            results.append((None, []))

    start_the_test(images, results, value)

postotak_c = round((correct_all / (correct_all + wrong_all)) * 100, 2)
postotak_w = round((wrong_all / (correct_all + wrong_all)) * 100, 2)
print(rezultati)
print(f"Correct: {correct_all} ({postotak_c}%)\nWrong: {wrong_all} ({postotak_w}%)")
cv2.destroyAllWindows()