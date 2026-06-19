from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
import os
import threading
import pyttsx3
from gesture_recognition import GestureRecognizer

app = Flask(__name__)

# Initialize the Gesture Recognizer
# It will load model/model.p if it exists, otherwise it will fall back to Heuristic mode.
recognizer = GestureRecognizer()

# Thread-safe lock for Text-to-Speech to prevent overlapping speech
tts_lock = threading.Lock()

def speak_text_worker(text):
    """
    Worker function to run in a separate thread.
    Initializes pyttsx3, speaks the text, and clean up.
    """
    with tts_lock:
        try:
            # Initialize COM and the speech engine locally in the thread
            engine = pyttsx3.init()
            
            # Set voice properties for a clear, natural voice
            engine.setProperty('rate', 150)    # Speed (words per minute)
            engine.setProperty('volume', 1.0)  # Volume level (0.0 to 1.0)
            
            # Use a female voice if available, otherwise default
            voices = engine.getProperty('voices')
            for voice in voices:
                if "female" in voice.name.lower() or "zira" in voice.name.lower():
                    engine.setProperty('voice', voice.id)
                    break
                    
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"Error in backend TTS thread: {e}")

@app.route('/')
def home():
    """Render landing home page."""
    return render_template('index.html')

@app.route('/recognition')
def recognition():
    """Render gesture recognition page."""
    return render_template('recognition.html')

@app.route('/about')
def about():
    """Render project documentation and about page."""
    return render_template('about.html')

@app.route('/contact')
def contact():
    """Render project contact page."""
    return render_template('contact.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    API endpoint to classify hand gestures.
    Accepts EITHER:
    {
        "landmarks": [{"x": 0.1, "y": 0.2, "z": 0.3}, ...],
        "mode": "heuristic" or "ml"
    }
    OR (fallback):
    {
        "image": "data:image/jpeg;base64,...",
        "mode": "heuristic" or "ml"
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No payload received'}), 400
            
        mode = data.get('mode', 'heuristic')
        model_loaded = recognizer.model is not None
        actual_mode = mode if (mode == 'heuristic' or model_loaded) else 'heuristic'
        
        # Scenario 1: Landmarks sent directly (Client-side MediaPipe)
        if 'landmarks' in data:
            landmarks_json = data['landmarks']
            if not landmarks_json or len(landmarks_json) < 21:
                return jsonify({'success': False, 'error': 'Invalid landmarks list'}), 400
                
            # Convert JSON format to coordinates list [[x, y, z], ...]
            lm_list = [[lm.get('x', 0.0), lm.get('y', 0.0), lm.get('z', 0.0)] for lm in landmarks_json]
            
            # Predict from coordinates
            prediction, confidence = recognizer.process_landmarks(lm_list, mode)
            
            response = {
                'success': True,
                'prediction': prediction,
                'confidence': float(confidence) if prediction else 0.0,
                'landmarks': landmarks_json,
                'mode_used': actual_mode,
                'model_available': model_loaded
            }
            return jsonify(response)
            
        # Scenario 2: Image frame sent (Server-side OpenCV & MediaPipe fallback)
        elif 'image' in data:
            image_data = data['image']
            
            # Strip metadata header if present
            if ',' in image_data:
                image_data = image_data.split(',')[1]
                
            # Decode base64 to image
            nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                return jsonify({'success': False, 'error': 'Could not decode image'}), 400
                
            # Process the frame
            prediction, confidence, landmarks, raw_lms = recognizer.process_frame(frame, mode)
            
            response = {
                'success': True,
                'prediction': prediction,
                'confidence': float(confidence) if prediction else 0.0,
                'landmarks': landmarks,
                'mode_used': actual_mode,
                'model_available': model_loaded
            }
            return jsonify(response)
            
        else:
            return jsonify({'success': False, 'error': 'Payload must contain landmarks or image'}), 400
            
    except Exception as e:
        print(f"API Predict Exception: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/speak', methods=['POST'])
def speak():
    """
    API endpoint to trigger text-to-speech on the server side.
    Expects JSON:
    {
        "text": "sentence to speak"
    }
    """
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'success': False, 'error': 'No text provided'}), 400
            
        text = data['text'].strip()
        if not text:
            return jsonify({'success': True, 'message': 'Empty text, skipped speaking.'})
            
        # Spin up a daemon thread to perform speech without blocking Flask's HTTP response
        tts_thread = threading.Thread(target=speak_text_worker, args=(text,))
        tts_thread.daemon = True
        tts_thread.start()
        
        return jsonify({'success': True, 'message': f"Speaking: '{text}'"})
        
    except Exception as e:
        print(f"API Speak Exception: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    # Ensure templates and static directories exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    os.makedirs('static/images', exist_ok=True)
    
    # Reload model on start if model exists
    recognizer.load_model()
    
    print("\n--- GestureTalk Flask Server Starting ---")
    print("Open http://127.0.0.1:5000 in your browser to view the application.")
    app.run(host='127.0.0.1', port=5000, debug=True)
