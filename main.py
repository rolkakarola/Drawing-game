import cv2
import numpy as np
import mediapipe as mp

# Open the camera
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Camera not accessible.")
    exit()

# Initialize variables
height, width = 480, 640  # Set the resolution manually
canvas = np.ones((height, width, 3), dtype="uint8") * 255  # White canvas to draw on
draw_color = (0, 102, 102)  # Color for drawing (teal)
brush_thickness = 2
previous_point = None

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils

# Function to adjust brightness and contrast
def adjust_brightness_contrast(image, brightness=0, contrast=0):
    beta = brightness
    alpha = contrast / 127 + 1
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return image

# Function to check if only the forefinger is up
def is_forefinger_up(hand_landmarks):
    # Get the y-coordinates of the fingertips and their lower joints
    tips = [
        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y,
        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y,
        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y,
        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y,
    ]
    lowers = [
        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y,
        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y,
        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y,
        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y,
    ]

    # Check if the index finger is raised (tip is above PIP)
    index_raised = tips[0] < lowers[0]

    # Check if other fingers are not raised
    others_down = all(tips[i] > lowers[i] for i in range(1, 4))

    return index_raised and others_down

# Start capturing frames from the camera
while True:
    ret, frame = cap.read()  # Read a frame from the camera

    if not ret:
        print("CAMERA ISN'T READY!")
        break

    frame = cv2.flip(frame, 1)  # Flip the frame horizontally

    # Adjust brightness and contrast
    frame = adjust_brightness_contrast(frame, brightness=-70, contrast=80)

    # Convert the frame to RGB (MediaPipe expects RGB input)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hands
    results = hands.process(rgb_frame)

    # Check if hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the coordinates of the index finger tip
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_tip_pos = (int(index_tip.x * width), int(index_tip.y * height))

            # Check if the forefinger is the only finger raised
            if is_forefinger_up(hand_landmarks):
                if previous_point is not None:
                    cv2.line(canvas, previous_point, index_tip_pos, draw_color, brush_thickness)
                previous_point = index_tip_pos
            else:
                previous_point = None

            # Draw a circle around the index tip
            cv2.circle(frame, index_tip_pos, 10, (0, 255, 0), -1)

    else:
        previous_point = None

    # Show the hand detection and drawing canvas
    cv2.imshow("HAND DETECTION", frame)
    cv2.imshow("DRAWING SCREEN", canvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        cv2.imwrite("hand_drawing.png", canvas)
        break

# Release the camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
