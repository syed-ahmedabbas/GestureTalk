# GestureTalk - AI Sign Language Translator

GestureTalk is a full-stack, accessibility-focused web application that translates hand gestures/sign language into text and speech in real-time. Designed to bridge the communication gap for deaf and mute individuals, it leverages computer vision and machine learning to interpret gestures through a standard webcam.

This project features a high-fidelity glassmorphic frontend, a robust dual-mode recognition pipeline (ML-based + Heuristic-based fallback), keyboard shortcuts, a dynamic history log, and a dual Speech Synthesis engine.

---

## Key Features

- **Real-Time Translation**: Interpret hand gestures at up to 30 frames per second using MediaPipe.
- **Dual Classification Modes**:
  - **Heuristic Mode**: Geometrical rule-based classification based on finger joint vectors. Works instantly out of the box with zero setup.
  - **Machine Learning Mode**: Scikit-learn Random Forest model trained on hand landmark coordinates. Includes a synthetic model generator to get started immediately, plus custom recording scripts.
- **Integrated Text-to-Speech (TTS)**: Dual implementation using client-side Web Speech API and backend `pyttsx3` (Windows-compatible).
- **Modern User Interface**: Responsive dark/light mode with premium glassmorphism styling, micro-animations, confidence gauges, and instant toast notifications.
- **Keyboard Shortcuts**: Control functions with keyboard inputs (`Enter` to speak, `Esc` to clear, `Space` for spaces, `Backspace` to delete).
- **History Log**: Track predictions and adjustments in a scrollable panel.

---

## Technology Stack

- **Frontend**: HTML5, CSS3 (Vanilla CSS with custom properties), Javascript (Vanilla ES6).
- **Backend**: Python 3.12, Flask.
- **Computer Vision / ML**: OpenCV-Python, MediaPipe (Hand Landmarks Mapping), Scikit-learn (Random Forest Classifier).
- **Audio Engine**: Pyttsx3, Web Speech Synthesis.

---

## Project Structure

```
GestureTalk/
│
├── app.py                      # Main Flask web server (APIs & routing)
├── gesture_recognition.py       # Core computer vision & classification module
├── collect_data.py             # CLI tool to record custom hand landmarks via webcam
├── train_model.py              # Script to train Scikit-learn model on custom data
├── generate_synthetic_model.py # Tool to auto-generate a baseline trained model
├── requirements.txt            # Python package dependencies
├── README.md                   # Project documentation
│
├── model/
│   └── model.p                 # Pickled Scikit-learn model file
│
├── templates/
│   ├── index.html              # Home landing page
│   ├── recognition.html        # Gesture recognition workspace
│   └── about.html              # System architecture & landmarks map
│
└── static/
    ├── css/
    │   └── style.css           # Glassmorphic layout & dark/light theme
    └── js/
        └── script.js           # Client-side webcam handler, API logic, state machine
```

---

## Setup & Installation

Follow these steps to run GestureTalk locally on your machine.

### Prerequisites
Make sure you have **Python 3.10+** installed on your system.

### 1. Clone the repository and navigate to the project directory:
```bash
cd GestureTalk
```

### 2. Set up a virtual environment (Recommended):
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install the dependencies:
```bash
pip install -r requirements.txt
```

### 4. Generate the baseline ML model (Required for ML Mode on first run):
Since a machine learning model is required to use ML Mode, you can automatically generate a baseline trained model using mathematically simulated landmarks:
```bash
python generate_synthetic_model.py
```
This generates `model/model.p` with high-accuracy parameters for the 8 default gestures.

### 5. Run the web application:
```bash
python app.py
```

Open your browser and navigate to **`http://127.0.0.1:5000`**.

---

## Supported Gestures

GestureTalk supports the following 8 gestures out of the box:

| Gesture | Meaning | Hand Shape Details |
| :---: | :--- | :--- |
| **✊** | **A** | A tight fist (all fingers closed). |
| **✋** | **B** | An open palm, flat and vertical (all fingers extended). |
| **🤏** | **C** | All fingers curved to form a C shape. |
| **☝️** | **L** | Index finger and thumb extended, others folded. |
| **🤙** | **Y** | Thumb and pinky extended, middle three fingers folded. |
| **✌️** | **SPACE** | Index and middle finger extended (Peace sign). Adds a space to the sentence. |
| **👍** | **DELETE** | Thumb extended, others folded (Thumbs up). Removes the last letter. |
| **🤘** | **CLEAR** | Index and pinky extended, middle two folded (Rock sign). Erases the text. |

---

## Custom Model Training (Record Your Own Signs!)

You can train a custom scikit-learn model using your own hand gestures.

### Step 1: Collect Custom Data
Run the data collector script:
```bash
python collect_data.py
```
1. Select the index of the gesture you want to record (0-7).
2. The webcam view will open. Position your hand and press **`s`** to start capturing.
3. Keep moving your hand slightly to record 100 frames from different angles.
4. Repeat this step for each gesture class you want to train.

### Step 2: Train the Classifier
Run the training script:
```bash
python train_model.py
```
This script reads all recorded `.npy` files from the `data/` folder, splits the data into training/testing sets, trains a Random Forest Classifier, outputs accuracy metrics, and saves the new model to `model/model.p`.

### Step 3: Run the Server
Restart the Flask server:
```bash
python app.py
```
The server will detect your custom `model.p` and load it automatically. Select **Machine Learning** mode in the configuration panel on the recognition workspace.

---

## Keyboard Shortcuts

When in the Recognition Workspace (and not typing inside settings menus), you can use the following shortcuts:

- **`Spacebar`**: Inserts a space character manually.
- **`Backspace`**: Deletes the last character in the sentence.
- **`Enter`**: Speaks the entire sentence aloud.
- **`Escape`**: Clears the entire recognized text.

---

## Project Status & Contributions

> [!NOTE]
> **GestureTalk is currently in its early phases of active development.** 
> Core systems (such as heuristic calculations, model interfaces, and keyboard actions) are functional, but we are actively working on expanding sign classes, optimizing tracking robustness, and improving translation flow.

We welcome and encourage community contributions! If you would like to help improve GestureTalk:
- **Report bugs or suggest features** by opening an issue.
- **Contribute code changes** by creating a fork, making your edits, and opening a pull request.
- **Expand the ML gesture library** by sharing custom landmark data frames.

Feel free to jump in and help build the future of accessible sign-to-speech communication!
