import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from datetime import datetime
import time
import pickle
import hashlib

# Configuration
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "trained", "best_model.h5")
USERS_DIR = os.path.join(PROJECT_ROOT, "data", "users")
os.makedirs(USERS_DIR, exist_ok=True)

# Authentication parameters
SEQUENCE_LENGTH = 5
MIN_CONFIDENCE = 0.75  # Lowered from 0.85 for better capture
SEQUENCE_THRESHOLD = 0.75
MAX_FAILED_ATTEMPTS = 3
HOLD_TIME = 1.0  # Seconds to hold gesture
COOLDOWN = 0.5  # Seconds between gestures


class GestureAuthenticator:
    def __init__(self):
        # Load trained model
        self.model = tf.keras.models.load_model(MODEL_PATH)
        self.class_names = ["fist", "palm", "ok", "rock", "salute", "bang"]

        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5)  # Lowered tracking confidence

        # Session variables
        self.current_gesture = None
        self.gesture_start_time = 0
        self.last_capture_time = 0
        self.failed_attempts = 0

    def preprocess_frame(self, frame):
        """Prepare frame for model prediction"""
        # Convert to RGB and resize
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (224, 224))
        normalized = resized / 255.0
        return np.expand_dims(normalized, axis=0)

    def predict_gesture(self, frame):
        """Predict gesture with enhanced reliability"""
        processed = self.preprocess_frame(frame)
        predictions = self.model.predict(processed, verbose=0)[0]

        # Get top 2 predictions
        top2_idx = np.argsort(predictions)[-2:][::-1]
        top1_conf = predictions[top2_idx[0]]
        top2_conf = predictions[top2_idx[1]]

        # Only confirm if significantly better than 2nd choice
        if top1_conf > MIN_CONFIDENCE and (top1_conf - top2_conf) > 0.15:
            return self.class_names[top2_idx[0]], top1_conf
        return None, 0

    def draw_gesture_feedback(self, frame, gesture, confidence):
        """Visual feedback for gesture capture"""
        h, w = frame.shape[:2]

        # Main instruction
        cv2.putText(frame, f"Show Gesture {len(self.current_sequence) + 1}/{SEQUENCE_LENGTH}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Current gesture detection
        if gesture:
            cv2.putText(frame, f"Detected: {gesture} ({confidence:.2f})",
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Progress bar
            elapsed = time.time() - self.gesture_start_time
            progress = min(1.0, elapsed / HOLD_TIME)
            cv2.rectangle(frame, (20, h - 30), (int(20 + (w - 40) * progress), h - 10),
                          (0, 255, 0), -1)
            cv2.rectangle(frame, (20, h - 30), (w - 20, h - 10), (255, 255, 255), 2)
        else:
            cv2.putText(frame, "Hold gesture clearly in frame",
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Sequence progress
        seq_text = " > ".join(self.current_sequence) if self.current_sequence else "None"
        cv2.putText(frame, f"Sequence: {seq_text}",
                    (20, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        return frame

    def capture_gesture_sequence(self, mode="verify"):
        """Improved gesture sequence capture"""
        self.current_sequence = []
        cap = cv2.VideoCapture(0)
        last_gesture = None

        while len(self.current_sequence) < SEQUENCE_LENGTH:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            current_time = time.time()

            # Detect gesture
            gesture, confidence = self.predict_gesture(frame)

            # Gesture confirmation logic
            if gesture and (current_time - self.last_capture_time) > COOLDOWN:
                if gesture != last_gesture:
                    self.gesture_start_time = current_time
                    last_gesture = gesture
                elif current_time - self.gesture_start_time > HOLD_TIME:
                    self.current_sequence.append(gesture)
                    print(f"Captured: {gesture} (Confidence: {confidence:.2f})")
                    self.last_capture_time = current_time
                    last_gesture = None

            # Display feedback
            frame = self.draw_gesture_feedback(frame, gesture, confidence)
            cv2.imshow(f"{mode.capitalize()} Gestures", frame)

            if cv2.waitKey(1) == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        return self.current_sequence

    def enroll_user(self, username):
        """User enrollment with reliable capture"""
        user_dir = os.path.join(USERS_DIR, username)
        os.makedirs(user_dir, exist_ok=True)

        print(f"\n=== Enrolling {username} ===")
        print(f"Perform your {SEQUENCE_LENGTH}-gesture sequence")
        print("Hold each gesture steady for 1 second")

        sequence = self.capture_gesture_sequence("enroll")

        if len(sequence) == SEQUENCE_LENGTH:
            template = {
                'sequence': sequence,
                'hash': hashlib.sha256(''.join(sequence).encode()).hexdigest(),
                'created_at': datetime.now().isoformat()
            }

            with open(os.path.join(user_dir, "template.pkl"), 'wb') as f:
                pickle.dump(template, f)

            print(f"\nEnrollment successful!")
            print(f"Your sequence: {' > '.join(sequence)}")
            return True
        else:
            print("\nEnrollment failed - incomplete sequence")
            return False

    def verify_user(self, username):
        """Verification with sequence matching"""
        user_dir = os.path.join(USERS_DIR, username)
        if not os.path.exists(user_dir):
            print(f"User {username} not found!")
            return False

        with open(os.path.join(user_dir, "template.pkl"), 'rb') as f:
            template = pickle.load(f)

        print(f"\n=== Verify {username} ===")
        print(f"Repeat your {SEQUENCE_LENGTH}-gesture sequence")

        input_sequence = self.capture_gesture_sequence("verify")

        if len(input_sequence) == SEQUENCE_LENGTH:
            # Calculate match score
            correct = sum(1 for i, gest in enumerate(input_sequence)
                          if gest == template['sequence'][i])
            score = correct / SEQUENCE_LENGTH

            if score >= SEQUENCE_THRESHOLD:
                print(f"\nAuthentication successful! (Score: {score:.2f})")
                self.failed_attempts = 0
                return True
            else:
                self.failed_attempts += 1
                print(f"\nAuthentication failed (Score: {score:.2f})")
                print(f"Expected: {' > '.join(template['sequence'])}")
                print(f"Received: {' > '.join(input_sequence)}")
                return False
        else:
            print("\nVerification incomplete")
            return False

    def run(self):
        """Main application loop"""
        while True:
            print("\n=== Gesture Authentication System ===")
            print("1. Enroll new user")
            print("2. Authenticate user")
            print("3. Exit")

            choice = input("Select option: ").strip()

            if choice == "1":
                username = input("Enter username: ").strip()
                self.enroll_user(username)
            elif choice == "2":
                username = input("Enter username: ").strip()
                if self.verify_user(username):
                    print("\nACCESS GRANTED")
                else:
                    if self.failed_attempts >= MAX_FAILED_ATTEMPTS:
                        print("\nSECURITY LOCK: Too many failed attempts!")
                        break
                    print("\nACCESS DENIED")
            elif choice == "3":
                break
            else:
                print("Invalid choice")


if __name__ == "__main__":
    auth = GestureAuthenticator()
    auth.run()