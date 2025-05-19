# ✋ Hand Gesture Authentication System

![Demo](docs/demo.gif)

> A secure, real-time authentication system that replaces passwords with custom hand gesture sequences.

---

## 🚧 Build Status

🛠️ This project is currently under active development.

---

## ✨ Features

* ✅ Supports six unique gestures: Fist, Palm, OK, Rock, Salute, Bang
* 🔒 Sequence-based authentication (5-gesture combinations)
* 🛡️ Anti-spoofing via liveness detection
* 📊 95%+ accuracy with proper training
* 🚀 Real-time performance (\~30 FPS on modern hardware)

---

## 📦 Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/rudrasatani13/hand-gesture-auth-system.git
   cd hand-gesture-auth-system
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ Usage

### 1. Data Collection

Capture gesture samples for training:

```bash
python src/data_collection/capture.py
```

Follow on-screen instructions for all 6 gestures.

### 2. Preprocessing

```bash
python src/preprocessing/processing.py
```

Applies augmentation and normalization.

### 3. Model Training

```bash
python src/training/train.py
```

Trains the CNN classifier.

### 4. Authentication

```bash
python src/auth/authenticator.py
```

Runs the real-time authentication interface.

---

## ✋ Gesture Guide

| Gesture | Emoji | Description                   |
| ------- | ----- | ----------------------------- |
| Fist    | ✊     | Closed hand                   |
| Palm    | 🖐️   | Open hand, palm facing camera |
| OK      | 👌    | Circle with index/thumb       |
| Rock    | 🤘    | Index and pinky extended      |
| Salute  | 🫡    | Hand to forehead              |
| Bang    | 🔫    | Finger gun                    |

➡️ See [docs/gesture\_guide.md](docs/gesture_guide.md) for more.

---

## ⚙️ Configuration

Edit `src/auth/config.py`:

```python
SEQUENCE_LENGTH = 5          # Gestures in auth sequence
MIN_CONFIDENCE = 0.75        # Model confidence threshold
HOLD_TIME = 1.0              # Time to hold each gesture (seconds)
MAX_FAILED_ATTEMPTS = 3      # Lockout after failed attempts
```

---

## 🧱 System Architecture

* **Capture**: Webcam input via MediaPipe
* **Process**: Background removal + data normalization
* **Classify**: CNN predicts gestures
* **Authenticate**: Matches input sequence with timing checks

---

## 📋 Requirements

* Python 3.8+
* TensorFlow >= 2.6.0
* OpenCV >= 4.5.0
* MediaPipe >= 0.8.9
* Webcam

---

## 🤝 Contributing

1. Fork the repository
2. Create a branch: `git checkout -b feature/AmazingFeature`
3. Commit your changes: `git commit -m 'Add AmazingFeature'`
4. Push to the branch: `git push origin feature/AmazingFeature`
5. Open a Pull Request

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more info.

---

## 📬 Contact

Your Name
📧 [rudrasatani@gmail.com](mailto:rudrasatani@gmail.com)
🔗 Project Link: [GitHub](https://github.com/rudrasatani13/hand-gesture-auth-system)

---

## 📁 Supporting Files

### .gitignore

```gitignore
# Data
data/raw/*
!data/raw/.gitkeep
data/processed/*
!data/processed/.gitkeep

# Models
models/trained/*
!models/trained/.gitkeep

# Python cache
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/
```

### requirements.txt

```
tensorflow>=2.6.0
opencv-python>=4.5.0
mediapipe>=0.8.9
numpy>=1.19.5
scikit-learn>=0.24.2
tqdm>=4.62.0
matplotlib>=3.4.3
imgaug>=0.4.0
```

### docs/gesture\_guide.md

#### Gesture Performance Guide

**Optimal Capture Tips**:

* Keep your hand centered and fill 60–80% of the frame.
* Maintain 1–2 feet distance.
* Use diffused front-facing light. Avoid backlight or shadows.

**Detailed Gesture Specs**:

* ✊ **Fist**: Fully closed, thumb in or out.
* 🖐️ **Palm**: Open hand, fingers spread slightly.
* 👌 **OK**: Thumb/index circle; other fingers extended.
* 🤘 **Rock**: Index/pinky out; middle/ring curled.
* 🫡 **Salute**: Flat hand to forehead.
* 🔫 **Bang**: Index forward, thumb up, others curled.

---

🎯 Designed for modularity, clarity, and ease of extension.
