import cv2
import mediapipe as mp

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)


def is_palm_open(hand_landmarks):
    # Indices for landmarks of fingers tips
    finger_tips_ids = [4, 8, 12, 16, 20]

    # Get the landmarks from the hand
    landmarks = hand_landmarks.landmark

    fingers = []

    for tip_index in finger_tips_ids:
        # Check if the finger is up
        if landmarks[tip_index].y < landmarks[tip_index - 2].y:
            fingers.append(1)  # Finger is up
        else:
            fingers.append(0)  # Finger is down

    return all(fingers)


def get_thumb_orientation(hand_landmarks, image_width, image_height):
    # Indices for landmarks of thumb tip and thumb base (CMC)
    thumb_tip_id = 4
    thumb_cmc_id = 1

    # Get the landmarks from the hand
    landmarks = hand_landmarks.landmark

    # Convert normalized coordinates to pixel coordinates
    thumb_tip = landmarks[thumb_tip_id]
    thumb_cmc = landmarks[thumb_cmc_id]

    tip_x = int(thumb_tip.x * image_width)
    tip_y = int(thumb_tip.y * image_height)
    cmc_x = int(thumb_cmc.x * image_width)
    cmc_y = int(thumb_cmc.y * image_height)

    # Calculate the vector components of the thumb direction
    vector_x = tip_x - cmc_x
    vector_y = tip_y - cmc_y

    # Determine thumb orientation based on vector direction
    if abs(vector_y) > abs(vector_x):
        if vector_y < 0:
            return "Thumb Up"
    else:
        if vector_x > 0:
            return "Thumb Right"
        else:
            return "Thumb Left"

    return None


while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip the image horizontally for a later selfie-view display
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and detect hands
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Check if the palm is open
            if is_palm_open(hand_landmarks):
                status = "Palm Open"
            else:
                # Check the thumb orientation
                image_height, image_width, _ = image.shape
                status = get_thumb_orientation(hand_landmarks, image_width, image_height) or "Unknown"

            cv2.putText(image, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Hand Tracking', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
