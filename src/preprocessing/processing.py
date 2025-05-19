import os
import cv2
import numpy as np
import mediapipe as mp
import random
from glob import glob
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "..", "..", "data", "raw")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "..", "data", "processed")
CLASSES = ["fist", "palm", "ok", "rock", "salute", "bang"]
IMAGE_SIZE = (224, 224)
AUGMENTATION_ENABLED = True
TRAIN_SPLIT = 0.8

mp_selfie_segmentation = mp.solutions.selfie_segmentation
mp_hands = mp.solutions.hands


def improved_remove_background(image):
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as segment:

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = segment.process(rgb_image)

        mask = results.segmentation_mask > 0.5
        mask = mask.astype(np.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        coords = cv2.findNonZero(mask)
        if coords is None:
            return image

        x, y, w, h = cv2.boundingRect(coords)

        pad = 20
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(image.shape[1], x + w + pad)
        y2 = min(image.shape[0], y + h + pad)

        cropped_img = image[y1:y2, x1:x2]
        cropped_mask = mask[y1:y2, x1:x2]

        cropped_mask_3d = np.dstack([cropped_mask] * 3)
        hand_only = cropped_img * cropped_mask_3d

        white_bg = np.ones_like(hand_only, dtype=np.uint8) * 255
        final = white_bg * (1 - cropped_mask_3d) + hand_only

        return final.astype(np.uint8)



def enhanced_normalization(image):

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)

    h, w = sharpened.shape[:2]
    scale = 224 / max(h, w)
    resized = cv2.resize(sharpened, (int(w * scale), int(h * scale)))

    pad_top = (224 - resized.shape[0]) // 2
    pad_bottom = 224 - resized.shape[0] - pad_top
    pad_left = (224 - resized.shape[1]) // 2
    pad_right = 224 - resized.shape[1] - pad_left
    padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right,
                                cv2.BORDER_CONSTANT, value=(255, 255, 255))

    normalized = padded / 255.0

    return normalized


def smart_augmentation(image):

    angle = random.uniform(-15, 15)
    rows, cols = image.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    rotated = cv2.warpAffine(image, M, (cols, rows), borderMode=cv2.BORDER_REFLECT)

    tx = random.uniform(-0.1, 0.1) * cols
    ty = random.uniform(-0.1, 0.1) * rows
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    translated = cv2.warpAffine(rotated, M, (cols, rows), borderMode=cv2.BORDER_REFLECT)

    if random.random() > 0.5:
        translated = cv2.flip(translated, 1)

    hsv = cv2.cvtColor(translated, cv2.COLOR_BGR2HSV)
    hsv = hsv.astype(np.float32)
    hsv[..., 1] = hsv[..., 1] * random.uniform(0.8, 1.2)
    hsv[..., 2] = hsv[..., 2] * random.uniform(0.8, 1.2)
    hsv = np.clip(hsv, 0, 255)
    final = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return final


def process_dataset():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    metadata = []

    print("\n=== Starting Enhanced Preprocessing ===")
    print(f"Input Directory: {INPUT_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")

    for gesture in CLASSES:
        gesture_dir = os.path.join(INPUT_DIR, gesture)
        if not os.path.exists(gesture_dir):
            print(f"[WARNING] Missing directory: {gesture_dir}")
            continue

        image_paths = glob(os.path.join(gesture_dir, "*.png"))
        if not image_paths:
            print(f"[WARNING] No images found in {gesture_dir}")
            continue

        print(f"\nProcessing {len(image_paths)} images for '{gesture}'")

        if len(image_paths) >= 2:
            train_paths, test_paths = train_test_split(
                image_paths, train_size=TRAIN_SPLIT, random_state=42)
        else:
            train_paths = image_paths
            test_paths = []
            print(f"[WARNING] Insufficient images for proper split in {gesture}")

        for phase, paths in [("train", train_paths), ("test", test_paths)]:
            if not paths:
                continue

            output_dir = os.path.join(OUTPUT_DIR, phase, gesture)
            os.makedirs(output_dir, exist_ok=True)

            for img_path in tqdm(paths, desc=f"{phase.upper()} - {gesture}"):
                try:

                    img = cv2.imread(img_path)
                    if img is None:
                        raise ValueError(f"Failed to read {img_path}")


                    bg_removed = improved_remove_background(img)
                    processed = enhanced_normalization(bg_removed)

                    filename = f"{gesture}_{os.path.basename(img_path)[:-4]}.npy"
                    np.save(os.path.join(output_dir, filename), processed)
                    metadata.append({
                        "filename": filename,
                        "path": os.path.join(phase, gesture, filename),
                        "label": gesture,
                        "phase": phase,
                        "augmented": False
                    })

                    if phase == "train" and AUGMENTATION_ENABLED:
                        augmented = smart_augmentation(bg_removed)
                        aug_filename = f"{gesture}_{os.path.basename(img_path)[:-4]}_aug.npy"
                        np.save(os.path.join(output_dir, aug_filename), augmented)
                        metadata.append({
                            "filename": aug_filename,
                            "path": os.path.join(phase, gesture, aug_filename),
                            "label": gesture,
                            "phase": phase,
                            "augmented": True
                        })

                except Exception as e:
                    print(f"\n[ERROR] Processing {img_path}: {str(e)}")
                    continue

    if metadata:
        df = pd.DataFrame(metadata)
        df.to_csv(os.path.join(OUTPUT_DIR, "metadata.csv"), index=False)
        print("\n=== Processing Complete ===")
        print(f"Total processed samples: {len(df)}")
        print(f"Saved metadata to: {os.path.join(OUTPUT_DIR, 'metadata.csv')}")

        visualize_enhanced_results()
    else:
        print("\n[ERROR] No images were processed successfully")


def visualize_enhanced_results():
    num_gestures = len(CLASSES)
    plt.figure(figsize=(15, 5 * num_gestures))

    examples = []
    for gesture in CLASSES:
        raw_images = glob(os.path.join(INPUT_DIR, gesture, "*.png"))
        proc_images = glob(os.path.join(OUTPUT_DIR, "train", gesture, "*.npy"))

        if raw_images and proc_images:
            examples.append((raw_images[0], proc_images[0], gesture))
        else:
            print(f"[WARNING] No images found for {gesture}")

    for idx, (raw_path, proc_path, gesture) in enumerate(examples):
        plt.subplot(num_gestures, 3, idx * 3 + 1)
        original = cv2.imread(raw_path)
        if original is not None:
            plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
            plt.title(f"Original: {gesture}")
            plt.axis('off')
        else:
            plt.text(0.5, 0.5, f"Failed to load\n{raw_path}", ha='center')
            plt.axis('off')

        plt.subplot(num_gestures, 3, idx * 3 + 2)
        if original is not None:
            bg_removed = improved_remove_background(original)
            plt.imshow(cv2.cvtColor(bg_removed, cv2.COLOR_BGR2RGB))
            plt.title("Background Removed")
            plt.axis('off')
        else:
            plt.text(0.5, 0.5, "N/A", ha='center')
            plt.axis('off')

        plt.subplot(num_gestures, 3, idx * 3 + 3)
        try:
            processed = np.load(proc_path)
            plt.imshow(processed)
            plt.title("Enhanced & Normalized")
            plt.axis('off')
        except:
            plt.text(0.5, 0.5, f"Failed to load\n{proc_path}", ha='center')
            plt.axis('off')

    plt.tight_layout()
    comparison_path = os.path.join(OUTPUT_DIR, "preprocessing_comparison.png")
    plt.savefig(comparison_path)
    print(f"\nSaved visualization to: {comparison_path}")
    plt.show()


if __name__ == "__main__":
    process_dataset()