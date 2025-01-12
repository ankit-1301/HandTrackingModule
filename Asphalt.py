import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Set webcam resolution
wCam = 640  # Increase resolution if needed
hCam = 480

# Open webcam and configure output
fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Use XVID for .avi or check supported codecs
out = cv2.VideoWriter('asphalt.avi', fourcc, 20.0, (wCam, hCam))

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)


def is_palm_open(hand_landmarks):
    finger_tips_ids = [4, 8, 12, 16, 20]
    landmarks = hand_landmarks.landmark
    fingers = [
        1 if landmarks[tip_index].y < landmarks[tip_index - 2].y else 0
        for tip_index in finger_tips_ids
    ]
    return all(fingers)


def get_thumb_orientation(hand_landmarks, image_width, image_height):
    thumb_tip_id, thumb_cmc_id = 4, 1
    landmarks = hand_landmarks.landmark
    thumb_tip = landmarks[thumb_tip_id]
    thumb_cmc = landmarks[thumb_cmc_id]

    tip_x, tip_y = int(thumb_tip.x * image_width), int(thumb_tip.y * image_height)
    cmc_x, cmc_y = int(thumb_cmc.x * image_width), int(thumb_cmc.y * image_height)

    vector_x, vector_y = tip_x - cmc_x, tip_y - cmc_y
    if abs(vector_y) > abs(vector_x):
        return "Boost" if vector_y < 0 else None
    return "Right" if vector_x > 0 else "Left"


current_key = None

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if is_palm_open(hand_landmarks):
                status, key_to_press = "Drift", 's'
            else:
                h, w, _ = image.shape
                status = get_thumb_orientation(hand_landmarks, w, h) or "Unknown"
                key_to_press = {'Boost': 'space', 'Left': 'a', 'Right': 'd'}.get(status)

            if key_to_press and key_to_press != current_key:
                if current_key:
                    pyautogui.keyUp(current_key)
                pyautogui.keyDown(key_to_press)
                current_key = key_to_press

            cv2.putText(image, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    out.write(image)  # Save frames to video
    cv2.imshow('Asphalt', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

if current_key:
    pyautogui.keyUp(current_key)

cap.release()
out.release()
cv2.destroyAllWindows()
