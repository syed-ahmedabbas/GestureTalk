import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def get_base_finger(base_x, base_y, spacing_x, length, is_extended):
    # A finger has 4 points: MCP, PIP, DIP, Tip
    # We will generate them relative to the MCP joint.
    # MCP is at (base_x, base_y)
    mcp = [base_x, base_y, 0.0]
    
    if is_extended:
        # Straight up
        pip = [base_x + spacing_x * 0.2, base_y - length * 0.35, -0.02]
        dip = [base_x + spacing_x * 0.4, base_y - length * 0.70, -0.04]
        tip = [base_x + spacing_x * 0.5, base_y - length * 1.0, -0.05]
    else:
        # Bending down into palm
        pip = [base_x + spacing_x * 0.1, base_y - length * 0.2, 0.05]
        dip = [base_x + spacing_x * 0.05, base_y - length * 0.1, 0.08]
        tip = [base_x, base_y - length * 0.05, 0.06]
        
    return [mcp, pip, dip, tip]

def get_base_thumb(is_extended):
    # Thumb: 1 (CMC), 2 (MCP), 3 (IP), 4 (Tip)
    cmc = [0.15, -0.15, 0.0]
    mcp = [0.22, -0.22, 0.0]
    
    if is_extended:
        # Outward to the right
        ip = [0.32, -0.25, -0.02]
        tip = [0.42, -0.28, -0.04]
    else:
        # Folded inside the palm
        ip = [0.18, -0.25, 0.04]
        tip = [0.12, -0.28, 0.05]
        
    return [cmc, mcp, ip, tip]

def generate_hand_sample(gesture_name, noise_level=0.03):
    # Base layout
    wrist = [0.0, 0.0, 0.0]
    
    # Check finger extension per gesture
    # A (Fist): all closed
    # B (Palm): all open
    # C (Curved): all partially bent (we can simulate by shortening the extension)
    # L: thumb, index open; middle, ring, pinky closed
    # Y: thumb, pinky open; index, middle, ring closed
    # SPACE (Peace): index, middle open; thumb, ring, pinky closed
    # DELETE (Thumbs up): thumb open; index, middle, ring, pinky closed
    # CLEAR (Rock on): index, pinky open; thumb, middle, ring closed
    
    # Default extensions
    thumb_ext = False
    index_ext = False
    middle_ext = False
    ring_ext = False
    pinky_ext = False
    is_c = False
    
    if gesture_name == 'A':
        pass
    elif gesture_name == 'B':
        thumb_ext = index_ext = middle_ext = ring_ext = pinky_ext = True
    elif gesture_name == 'C':
        is_c = True
    elif gesture_name == 'L':
        thumb_ext = index_ext = True
    elif gesture_name == 'Y':
        thumb_ext = pinky_ext = True
    elif gesture_name == 'SPACE':
        index_ext = middle_ext = True
    elif gesture_name == 'DELETE':
        thumb_ext = True
    elif gesture_name == 'CLEAR':
        index_ext = pinky_ext = True
        
    # Generate points
    landmarks = [wrist]
    
    # Thumb
    landmarks.extend(get_base_thumb(thumb_ext))
    
    # Fingers: MCP, PIP, DIP, Tip
    # spacing_x is used to flare fingers out
    landmarks.extend(get_base_finger(0.1, -0.3, 0.05, 0.45, index_ext))   # Index
    landmarks.extend(get_base_finger(0.0, -0.32, 0.0, 0.48, middle_ext))  # Middle
    landmarks.extend(get_base_finger(-0.1, -0.3, -0.05, 0.45, ring_ext))  # Ring
    landmarks.extend(get_base_finger(-0.2, -0.26, -0.1, 0.38, pinky_ext)) # Pinky
    
    if is_c:
        # For C: scale down y coordinates of tips slightly to make it curved
        # index: 5,6,7,8 -> tip (8)
        # middle: 9,10,11,12 -> tip (12)
        # ring: 13,14,15,16 -> tip (16)
        # pinky: 17,18,19,20 -> tip (20)
        # thumb: 1,2,3,4 -> tip (4)
        # We simulate "C" by bending all fingers to a middle state
        landmarks = [wrist]
        # Thumb: slightly open
        landmarks.extend(get_base_thumb(True))
        landmarks[3] = [0.25, -0.20, 0.05] # IP
        landmarks[4] = [0.28, -0.15, 0.08] # Tip
        
        # Fingers: MCP, PIP, DIP, Tip curved
        # Tip is bent back down towards wrist
        for base_x, spacing_x, length in [(0.1, 0.05, 0.45), (0.0, 0.0, 0.48), (-0.1, -0.05, 0.45), (-0.2, -0.1, 0.38)]:
            mcp = [base_x, -0.3, 0.0]
            pip = [base_x + spacing_x * 0.25, -0.3 - length * 0.3, 0.08]
            dip = [base_x + spacing_x * 0.1, -0.3 - length * 0.45, 0.12]
            tip = [base_x, -0.3 - length * 0.35, 0.10]
            landmarks.extend([mcp, pip, dip, tip])
            
    # Add noise
    landmarks = np.array(landmarks)
    noise = np.random.normal(0, noise_level, landmarks.shape)
    # Don't add noise to the wrist base (keep it at exactly 0.0, 0.0, 0.0 for easier wrist-centering)
    noise[0] = [0.0, 0.0, 0.0]
    landmarks = landmarks + noise
    
    # Normalize
    wrist_pt = landmarks[0]
    temp_lms = landmarks - wrist_pt
    flat_lms = temp_lms.flatten()
    max_val = np.max(np.abs(flat_lms))
    if max_val > 0:
        flat_lms = flat_lms / max_val
        
    return flat_lms

def create_synthetic_dataset(samples_per_class=300):
    gestures = ['A', 'B', 'C', 'L', 'Y', 'SPACE', 'DELETE', 'CLEAR']
    gesture_labels = {g: i for i, g in enumerate(gestures)}
    
    X = []
    y = []
    
    for g in gestures:
        label = gesture_labels[g]
        for _ in range(samples_per_class):
            sample = generate_hand_sample(g)
            X.append(sample)
            y.append(label)
            
    return np.array(X), np.array(y)

def main():
    print("Generating synthetic hand landmark dataset...")
    X, y = create_synthetic_dataset(samples_per_class=400)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Random Forest Classifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy on test split: {accuracy * 100:.2f}%")
    
    os.makedirs("model", exist_ok=True)
    model_path = "model/model.p"
    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)
        
    print(f"Baseline model successfully generated and saved to {model_path}!")

if __name__ == "__main__":
    main()
