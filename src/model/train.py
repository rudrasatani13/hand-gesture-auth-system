import os
import ssl
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from sklearn.model_selection import train_test_split

# ================== CONFIGURATION ==================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "trained")
os.makedirs(MODEL_DIR, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50
CLASSES = ["fist", "palm", "ok", "rock", "salute", "bang"]
NUM_CLASSES = len(CLASSES)


# ================== DATA LOADING ==================
def load_data(data_dir, classes):
    X_train, y_train, X_test, y_test = [], [], [], []

    print("\n[INFO] Loading dataset...")

    for split in ['train', 'test']:
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"Directory not found: {split_dir}")

        for class_idx, class_name in enumerate(classes):
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.exists(class_dir):
                raise FileNotFoundError(f"Class directory not found: {class_dir}")

            npy_files = [f for f in os.listdir(class_dir) if f.endswith('.npy')]
            if not npy_files:
                print(f"[WARNING] No .npy files found in {class_dir}")
                continue

            print(f"Loading {len(npy_files)} samples from {class_dir}")

            for npy_file in npy_files:
                try:
                    img = np.load(os.path.join(class_dir, npy_file))
                    
                    # Ensure proper shape and type
                    img = img.astype(np.float32)  # Convert to float32
                    
                    # Resize if necessary
                    if img.shape[:2] != (224, 224):
                        img = tf.image.resize(img, (224, 224)).numpy()
                    
                    # Handle different channel configurations
                    if len(img.shape) == 2:
                        img = np.stack((img,) * 3, axis=-1)
                    elif len(img.shape) == 3:
                        if img.shape[-1] == 1:
                            img = np.repeat(img, 3, axis=-1)
                        elif img.shape[-1] != 3:
                            raise ValueError(f"Unexpected number of channels: {img.shape[-1]}")
                    
                    # Ensure the image has the correct shape
                    if img.shape != (224, 224, 3):
                        raise ValueError(f"Invalid image shape after processing: {img.shape}")

                    if split == 'train':
                        X_train.append(img)
                        y_train.append(class_idx)
                    else:
                        X_test.append(img)
                        y_test.append(class_idx)
                except Exception as e:
                    print(f"[ERROR] Failed to load {npy_file}: {str(e)}")
                    continue

    # Convert to numpy arrays with explicit dtype
    X_train = np.array(X_train, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)
    y_test = np.array(y_test, dtype=np.int32)

    print(f"\nLoaded data shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    
    return X_train, y_train, X_test, y_test


# ================== MODEL ARCHITECTURE ==================
def build_model():
    ssl._create_default_https_context = ssl._create_unverified_context

    base_model = MobileNetV2(include_top=False, input_shape=(224, 224, 3), weights='imagenet')
    base_model.trainable = False  # Freeze base layers

    inputs = Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# ================== VISUALIZATION ==================
def plot_sample_images(X, y, title):
    plt.figure(figsize=(15, 5))
    for i in range(min(5, len(X))):
        plt.subplot(1, 5, i + 1)
        plt.imshow(X[i].astype(np.uint8))
        plt.title(f"{CLASSES[y[i]]}")
        plt.axis('off')
    plt.suptitle(title)
    plt.show()


def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'training_history.png'))
    plt.show()


# ================== MAIN TRAINING ==================
def main():
    try:
        # Load data
        X_train, y_train, X_test, y_test = load_data(DATA_DIR, CLASSES)

        # Convert labels to categorical
        y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
        y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size=0.2,
            stratify=np.argmax(y_train, axis=1),
            random_state=42
        )
        # ===== DATA VERIFICATION =====
        print("\n=== DATA SUMMARY ===")
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")

        # Show sample images
        plot_sample_images(X_train, np.argmax(y_train, axis=1), "Training Samples")
        plot_sample_images(X_test, np.argmax(y_test, axis=1), "Test Samples")

        # ===== CLASS WEIGHTS =====
        train_classes = np.argmax(y_train, axis=1)
        class_weights = compute_class_weight('balanced', classes=np.unique(train_classes), y=train_classes)
        class_weights = dict(enumerate(class_weights))
        print("\nClass weights:", class_weights)

        model = build_model()
        model.summary()

        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint(
                os.path.join(MODEL_DIR, 'best_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max'
            ),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
        ]

        # ===== DATA AUGMENTATION =====
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.05,
            height_shift_range=0.05,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        # Train model
        print("\n=== TRAINING STARTED ===")
        history = model.fit(
            train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
            steps_per_epoch=len(X_train) // BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            class_weight=class_weights
        )

        # Save model
        model.save(os.path.join(MODEL_DIR, 'final_model.h5'))
        print(f"\nModel saved to {MODEL_DIR}")

        # Evaluation
        plot_training_history(history)

        y_pred = np.argmax(model.predict(X_test), axis=1)
        y_true = np.argmax(y_test, axis=1)

        print("\n=== CLASSIFICATION REPORT ===")
        print(classification_report(y_true, y_pred, target_names=CLASSES))

        # Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=CLASSES, yticklabels=CLASSES)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(os.path.join(MODEL_DIR, 'confusion_matrix.png'))
        plt.show()

    except Exception as e:
        print(f"\n[ERROR] Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()