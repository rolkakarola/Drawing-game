import pickle
import cv2
import mediapipe as mp
import numpy as np

# Main code + interference classifier

class HandGestureRecognition:
    def __init__(self, model_path, labels_dict, camera_index=1, detection_confidence=0.3):
        self.model = self.load_model(model_path)
        self.labels_dict = labels_dict
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            print("Error: Camera not accessible.")
            exit()

        self.canvas = None
        self.draw_color = (0, 0, 0)  # Color for drawing 
        self.previous_point = None
        self.brush_thickness = 5
        
        # Initialize MediaPipe hands
        self.mp_hands = mp.solutions.hands # hand tracking
        self.hands = self.mp_hands.Hands(min_detection_confidence=detection_confidence, min_tracking_confidence=0.8)
        self.mp_drawing = mp.solutions.drawing_utils # hand landmarks and connections
        
    def load_model(self, model_path):
        model_dict = pickle.load(open(model_path, 'rb'))
        return model_dict['model']

    def adjust_brightness_contrast(self, image):
        """Adjust the brightness and contrast of the image."""
        alpha = 1.0  # contrast factor (not used, fixed)
        beta = -70  # brightness adjustment (fixed for this code)
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    def is_forefinger_up(self, hand_landmarks):
        """Check if only the forefinger is raised."""
        tips = [ 
            hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y,
            hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y,
            hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP].y,
            hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP].y,
        ] # y-coordinates of the tips of the index, middle, ring, and pinky fingers
        lowers = [
            hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP].y,
            hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y,
            hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_PIP].y,
            hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_PIP].y,
        ] # y-coordinates of the second joint
        
        # Check if the index finger is raised (tip is above PIP)
        index_raised = tips[0] < lowers[0]

        # Check if other fingers are not raised
        others_down = all(tips[i] > lowers[i] for i in range(1, 4))

        return index_raised and others_down

    def process_frame(self, frame):
        """Process the frame for hand detection and drawing."""
        frame = cv2.flip(frame, 1)  # Flip the frame horizontally
        frame = self.adjust_brightness_contrast(frame)

        # Initialize canvas with the same size as the frame
        if self.canvas is None:
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

                # Check if the forefinger is the only finger raised
                if self.is_forefinger_up(hand_landmarks):
                    if self.previous_point is not None:
                        cv2.line(self.canvas, self.previous_point, index_tip_pos, self.draw_color, self.brush_thickness)
                    self.previous_point = index_tip_pos
                else:
                    self.previous_point = None

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
                elif predicted_character == 'clean screen':
                    # Clear the canvas (clean screen)
                    self.canvas = np.ones_like(frame) * 255
                    print("Screen cleaned")
                elif predicted_character == 'change width':
                    # Adjust the brush thickness
                    self.brush_thickness = np.random.randint(1, 10)
                    print(f"Brush width changed to {self.brush_thickness}")

                # Display predicted gesture on frame
                #x1 = int(min(x_) * W) - 10
                #y1 = int(min(y_) * H) - 10
                #x2 = int(max(x_) * W) - 10
                #y2 = int(max(y_) * H) - 10

                #displaying the rectangle around fingers
                #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_character, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3, cv2.LINE_AA)

                # Draw a circle around the index tip
                cv2.circle(frame, index_tip_pos, 10, (0, 255, 0), -1)

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
labels_dict = {0: 'change color', 1: 'clean screen', 2: 'change width'}
hand_gesture_recognition = HandGestureRecognition('./model.p', labels_dict)
hand_gesture_recognition.run()
