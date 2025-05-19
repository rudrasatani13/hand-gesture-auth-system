import cv2
import mediapipe as mp
import os
import time
from datetime import datetime

MIN_RESOLUTION = (640, 480)
SAVE_MARGIN = 0.2

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"[INFO] Created folder: {folder_path}")
    else:
        print(f"[INFO] Folder already exists: {folder_path}")

def crop_with_margin(image, bbox, margin=0.2):
    h, w, _ = image.shape
    x_min, y_min, x_max, y_max = bbox

    x_margin = int((x_max - x_min) * margin)
    y_margin = int((y_max - y_min) * margin)

    x_min = max(x_min - x_margin, 0)
    y_min = max(y_min - y_margin, 0)
    x_max = min(x_max + x_margin, w)
    y_max = min(y_max + y_margin, h)

    return image[y_min:y_max, x_min:x_max]

def get_hand_bbox(hand_landmarks, image_shape):
    h, w, _ = image_shape
    x_coords = [int(landmark.x * w) for landmark in hand_landmarks.landmark]
    y_coords = [int(landmark.y * h) for landmark in hand_landmarks.landmark]
    return min(x_coords), min(y_coords), max(x_coords), max(y_coords)


def main():
    gesture_type = input("Enter gesture type (e.g., 'fist', 'palm', 'peace'): ").strip().lower()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "../.."))

    save_folder = os.path.join(project_root, "data", "raw", gesture_type)
    create_folder_if_not_exists(save_folder)

    print(f"[INFO] Ready to save gesture data in: {save_folder}")
    print(f"[DEBUG] Project root: {project_root}")
    print(f"[DEBUG] Full save path: {save_folder}")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, MIN_RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, MIN_RESOLUTION[1])

    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7) as hands:

        print("[INFO] Starting hand gesture capture. Press 's' to save, 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to grab frame.")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                bbox = get_hand_bbox(results.multi_hand_landmarks[0], frame.shape)

                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                              (0, 255, 0), 2)
            else:
                bbox = None

            cv2.imshow("Hand Gesture Capture", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("[INFO] Quitting...")
                break
            elif key == ord('s'):
                if bbox:
                    cropped_img = crop_with_margin(frame, bbox, margin=SAVE_MARGIN)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    filename = f"{gesture_type}_{timestamp}.png"
                    filepath = os.path.join(save_folder, filename)
                    cv2.imwrite(filepath, cropped_img)
                    print(f"[CAPTURED] Saved: {filepath}")

                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                                  (0, 0, 255), 4)
                    cv2.imshow("Hand Gesture Capture", frame)
                    cv2.waitKey(150)
                else:
                    print("[WARNING] No hand detected. Cannot save frame.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
