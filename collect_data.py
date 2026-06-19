import cv2
import mediapipe as mp
import numpy as np
import os
import time

def main():
    # Setup MediaPipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    mp_draw = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # Gestures
    gestures = ['A', 'B', 'C', 'L', 'Y', 'SPACE', 'DELETE', 'CLEAR']
    print("Available Gestures:")
    for idx, gesture in enumerate(gestures):
        print(f"  {idx}: {gesture}")

    # Prompt user for class selection
    while True:
        try:
            class_idx = int(input("\nEnter the number (0-7) of the gesture you want to collect: "))
            if 0 <= class_idx < len(gestures):
                break
            print("Invalid index. Choose between 0 and 7.")
        except ValueError:
            print("Please enter a valid integer.")

    active_gesture = gestures[class_idx]
    print(f"\nPreparing to collect data for gesture: '{active_gesture}'")
    print("Instructions:")
    print("  1. Position your hand in front of the camera.")
    print("  2. Press 's' to start collecting (it will capture 100 samples).")
    print("  3. Keep moving your hand slightly to capture different angles/distances.")
    print("  4. Press 'q' to quit.")

    # Create data directory
    data_dir = "data"
    os.makedirs(os.path.join(data_dir, str(class_idx)), exist_ok=True)

    # Start webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    count = 0
    collecting = False
    max_samples = 100
    
    print("\nWebcam opening...")
    time.sleep(1)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Flip horizontally for mirroring
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # Process MediaPipe landmarks
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        # UI Overlay
        status_text = f"Gesture: {active_gesture} | Samples: {count}/{max_samples}"
        color = (0, 0, 255)
        if collecting:
            status_text += " | COLLECTING..."
            color = (0, 255, 0)
            
        cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, "Press 's' to Start | 'q' to Quit", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Draw hand landmarks
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            
            # Save landmark coordinates if collecting
            if collecting:
                lm_list = []
                for lm in hand_landmarks.landmark:
                    lm_list.append([lm.x, lm.y, lm.z])
                
                # Wrist centering
                wrist = lm_list[0]
                temp_lms = []
                for lm in lm_list:
                    temp_lms.append([lm[0] - wrist[0], lm[1] - wrist[1], lm[2] - wrist[2]])
                    
                # Flatten
                flat_lms = [coord for lm in temp_lms for coord in lm]
                
                # Scale normalization
                max_val = max(map(abs, flat_lms))
                if max_val > 0:
                    flat_lms = [val / max_val for val in flat_lms]
                
                # Save to disk
                np.save(os.path.join(data_dir, str(class_idx), f"{int(time.time() * 1000)}.npy"), flat_lms)
                count += 1
                
                if count >= max_samples:
                    print(f"\nFinished collecting {max_samples} samples for '{active_gesture}'!")
                    collecting = False
                    count = 0

        cv2.imshow("GestureTalk - Data Collector", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') and not collecting:
            collecting = True
            count = 0
            print("Starting collection...")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Data collection closed.")

if __name__ == "__main__":
    main()
