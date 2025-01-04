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
height, width = 480, 640  # Set the resolution manually, or you can get it dynamically
canvas = np.ones((height, width, 3), dtype="uint8") * 255  # White canvas to draw on
draw_color = (0, 0, 255)  # Red color for drawing
brush_thickness = 5
previous_point = None
is_drawing = False
hand_detected = False

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

# Function to calculate the distance between two points
def calculate_distance(point1, point2):
    return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

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

            # Get the coordinates of the thumb tip and index tip
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip_pos = (int(thumb_tip.x * width), int(thumb_tip.y * height))
            index_tip_pos = (int(index_tip.x * width), int(index_tip.y * height))

            # Calculate the distance between the thumb and index finger
            distance = calculate_distance(thumb_tip_pos, index_tip_pos)

            # If the distance is small, the hand is closed, and we start drawing
            if distance < 50 and not is_drawing:
                print("Hand is closed, START DRAWING!")
                is_drawing = True
                previous_point = None  # Reset previous point when starting drawing

            # If the distance is large, the hand is open, and we stop drawing
            elif distance > 100 and is_drawing:
                print("Hand is open, STOP DRAWING.")
                is_drawing = False
                previous_point = None  # Reset previous point when stopping drawing

            # If drawing mode is active, draw on canvas
            if is_drawing:
                if previous_point is not None:
                    cv2.line(canvas, previous_point, thumb_tip_pos, draw_color, brush_thickness)
                previous_point = thumb_tip_pos

            # Draw a circle around the thumb tip
            cv2.circle(frame, thumb_tip_pos, 10, (0, 255, 0), -1)

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
