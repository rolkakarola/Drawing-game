import pickle
import cv2
import mediapipe as mp
import numpy as np

# Main code + interference classifier
class HandGestureRecognition:
    def __init__(self, model_path, labels_dict, camera_index=1, detection_confidence=0.3, smoothing_factor=0.5, min_move_threshold=5):
        # Load model and initialize camera and other variables
        self.model = self.load_model(model_path)
        self.labels_dict = labels_dict
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            print("Error: Camera not accessible.")
            exit()

        self.canvas = None
        self.draw_color = (0, 0, 0)  # Color for drawing
        self.previous_point = None
        self.smoothed_point = None  # For smoothed position
        self.brush_thickness = 5
        self.smoothing_factor = smoothing_factor  # Smoothing factor (0 to 1)
        self.min_move_threshold = min_move_threshold  # Minimum distance to consider for drawing
        
        # Initialize MediaPipe hands
        self.mp_hands = mp.solutions.hands  # hand tracking
        self.hands = self.mp_hands.Hands(min_detection_confidence=detection_confidence, min_tracking_confidence=0.8)
        self.mp_drawing = mp.solutions.drawing_utils  # hand landmarks and connections
        
    def load_model(self, model_path):
        model_dict = pickle.load(open(model_path, 'rb'))
        return model_dict['model']

    def adjust_brightness_contrast(self, image):
        """Adjust the brightness and contrast of the image."""
        alpha = 1.0  # contrast factor (not used, fixed)
        beta = -70  # brightness adjustment (fixed for this code)
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    def process_frame(self, frame):
        """Process the frame for hand detection and drawing."""
        frame = cv2.flip(frame, 1)  # Flip the frame horizontally
        frame = self.adjust_brightness_contrast(frame)

        # Initialize canvas with the same size as the frame
        if self.canvas is None or self.canvas.shape != frame.shape:
            self.canvas = np.ones_like(frame) * 255  # White canvas to draw on

        # Convert the frame to RGB (MediaPipe expects RGB input)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        # Check if hands are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on the frame
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                # Get the coordinates of the index finger tip
                index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_tip_pos = (int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0]))

                # Apply smoothing to the fingertip position
                if self.smoothed_point is None:
                    self.smoothed_point = index_tip_pos  # Initialize with the first point
                else:
                    smoothed_x = int(self.smoothing_factor * index_tip_pos[0] + (1 - self.smoothing_factor) * self.smoothed_point[0])
                    smoothed_y = int(self.smoothing_factor * index_tip_pos[1] + (1 - self.smoothing_factor) * self.smoothed_point[1])
                    self.smoothed_point = (smoothed_x, smoothed_y)

                # Predict gesture using the model
                data_aux = []
                x_ = []
                y_ = []
                H, W, _ = frame.shape

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

                prediction = self.model.predict([np.asarray(data_aux)])
                predicted_character = self.labels_dict[int(prediction[0])]

                # Handle the actions based on the predicted gesture
                if predicted_character == 'change color':
                    # Change drawing color
                    self.draw_color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
                    print(f"Color changed to {self.draw_color}")
                elif predicted_character == 'clear screen':
                    # Clear the canvas (clean screen)
                    self.canvas.fill(255)  # Reset canvas to white
                    print("Screen cleaned")
        
                elif predicted_character == 'draw':
                    # Draw when the predicted gesture is "draw"
                    if self.previous_point is not None:
                        # Only draw if the movement is greater than a threshold to avoid jitter
                        distance = np.sqrt((self.smoothed_point[0] - self.previous_point[0]) ** 2 + (self.smoothed_point[1] - self.previous_point[1]) ** 2)
                        if distance > self.min_move_threshold:
                            cv2.line(self.canvas, self.previous_point, self.smoothed_point, self.draw_color, self.brush_thickness)
                    self.previous_point = self.smoothed_point
                else:
                    self.previous_point = None  # Reset drawing if no valid gesture is detected

                # Display predicted gesture on frame
                cv2.putText(frame, predicted_character, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3, cv2.LINE_AA)

                # Draw a circle around the index tip
                cv2.circle(frame, self.smoothed_point, 10, (0, 255, 0), -1)

        # Combine the frame and the canvas
        combined = cv2.addWeighted(frame, 0.5, self.canvas, 0.5, 0)

        return combined

    def run(self):
        """Start the camera capture and process frames."""
        while True:
            ret, frame = self.cap.read()  # Read a frame from the camera
            if not ret:
                print("CAMERA ISN'T READY!")
                break

            combined = self.process_frame(frame)

            # Show the combined result
            cv2.imshow("Hand Gesture Recognition and Drawing", combined)

            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                break

        # Release the camera and close OpenCV windows
        self.cap.release()
        cv2.destroyAllWindows()


# Usage
labels_dict = {0: 'draw', 1: 'change color', 2: 'clear screen'}
hand_gesture_recognition = HandGestureRecognition('./model.p', labels_dict)
hand_gesture_recognition.run()
