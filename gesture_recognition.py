import cv2
import numpy as np
import pickle
import os
import math

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except Exception as e:
    print(f"Warning: MediaPipe Python library could not be fully loaded: {e}")
    print("MediaPipe server-side image processing will be unavailable. Client-side landmarks will be used.")
    MEDIAPIPE_AVAILABLE = False

class GestureRecognizer:
    def __init__(self, model_path="model/model.p"):
        # Initialize MediaPipe Hands if available
        self.hands = None
        if MEDIAPIPE_AVAILABLE:
            try:
                self.mp_hands = mp.solutions.hands
                self.hands = self.mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=1,
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.5
                )
                self.mp_draw = mp.solutions.drawing_utils
                self.mp_drawing_styles = mp.solutions.drawing_styles
            except AttributeError as ae:
                print(f"Warning: MediaPipe solutions missing attributes (expected on Python 3.13/3.14): {ae}")
                self.hands = None
            except Exception as e:
                print(f"Error initializing MediaPipe C++ bindings: {e}")
                self.hands = None
        
        # Load ML model
        self.model_path = model_path
        self.model = None
        self.load_model()
        
        # Gesture mappings
        self.labels_map = {
            0: 'A',
            1: 'B',
            2: 'C',
            3: 'L',
            4: 'Y',
            5: 'SPACE',
            6: 'DELETE',
            7: 'CLEAR'
        }

    def load_model(self):
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                print(f"ML Model loaded successfully from {self.model_path}")
            except Exception as e:
                print(f"Error loading ML model: {e}")
                self.model = None
        else:
            print(f"ML Model file not found at {self.model_path}. Will fall back to Heuristic Mode.")

    def normalize_landmarks(self, lm_list):
        """
        Normalize landmarks to be translation-invariant (wrist-centered)
        and scale-invariant (scaled relative to maximum dimension).
        Returns a flat list of 63 values.
        """
        # lm_list is a list of 21 [x, y, z] lists
        wrist = lm_list[0]
        temp_lms = []
        for lm in lm_list:
            temp_lms.append([lm[0] - wrist[0], lm[1] - wrist[1], lm[2] - wrist[2]])
            
        # Flatten
        flat_lms = [coord for lm in temp_lms for coord in lm]
        
        # Scale normalization: divide by maximum absolute value
        max_val = max(map(abs, flat_lms))
        if max_val > 0:
            flat_lms = [val / max_val for val in flat_lms]
            
        return flat_lms

    def get_distance(self, p1, p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)

    def classify_heuristic(self, lm_list):
        """
        Rule-based classifier evaluating finger extension states and distances.
        Supports: A, B, C, L, Y, SPACE, DELETE, CLEAR
        """
        # Extract finger tips and joints
        # 0: Wrist, 4: Thumb Tip, 8: Index Tip, 12: Middle Tip, 16: Ring Tip, 20: Pinky Tip
        # Joints: 3 (Thumb IP), 6 (Index PIP), 10 (Middle PIP), 14 (Ring PIP), 18 (Pinky PIP)
        # Knuckles: 2 (Thumb MCP), 5 (Index MCP), 9 (Middle MCP), 13 (Ring MCP), 17 (Pinky MCP)
        
        # 1. Determine finger states (extended vs folded) using relative y-positions
        index_open = lm_list[8][1] < lm_list[6][1]
        middle_open = lm_list[12][1] < lm_list[10][1]
        ring_open = lm_list[16][1] < lm_list[14][1]
        pinky_open = lm_list[20][1] < lm_list[18][1]
        
        # For thumb: check distance from thumb tip to index base (mcp)
        # and thumb tip to pinky base. If it's extended outward, it will be far from middle/index.
        thumb_open = self.get_distance(lm_list[4], lm_list[9]) > self.get_distance(lm_list[3], lm_list[9]) * 1.15
        
        # Calculate palm size for relative distance scaling
        palm_size = self.get_distance(lm_list[0], lm_list[9])
        
        # Check if fingers are bent in a curved shape (for 'C')
        # Tips are not open, but also not fully closed (not below MCPs).
        # Specifically, check if the distance from wrist to finger tips is medium.
        # Flat hand: tip is far. Fist: tip is very close. Curved: in between.
        tip_dists = [self.get_distance(lm_list[i], lm_list[0]) for i in [8, 12, 16, 20]]
        avg_tip_dist = sum(tip_dists) / len(tip_dists)
        
        # 2. Classification Rules
        # A (Fist): All fingers closed
        if not index_open and not middle_open and not ring_open and not pinky_open:
            if thumb_open:
                return 'DELETE', 0.9  # Thumbs Up
            else:
                return 'A', 0.95
                
        # B (Palm): All fingers open, and straight
        if index_open and middle_open and ring_open and pinky_open:
            # If thumb is also open, and hand is flat
            # Check if hand is horizontal for SPACE: if the vector from index base (5) to pinky base (17) is mostly vertical
            # but usually let's distinguish B and SPACE. B is flat vertical/diagonal hand.
            return 'B', 0.95
            
        # L: Thumb and index open, others closed
        if thumb_open and index_open and not middle_open and not ring_open and not pinky_open:
            return 'L', 0.95
            
        # Y: Thumb and pinky open, others closed
        if thumb_open and pinky_open and not index_open and not middle_open and not ring_open:
            return 'Y', 0.95
            
        # SPACE (Peace sign): Index and middle open, others closed
        if index_open and middle_open and not ring_open and not pinky_open:
            return 'SPACE', 0.90
            
        # CLEAR (Rock on): Index and pinky open, others closed
        if index_open and pinky_open and not middle_open and not ring_open:
            return 'CLEAR', 0.90
            
        # C (Curved hand): All fingers partially bent
        # If all fingers are closed but tips are further from wrist than a tight fist (palm_size * 0.8)
        # and closer than a straight hand (palm_size * 1.5)
        is_curved = True
        for i in [8, 12, 16, 20]:
            # Tip is below PIP but tip y-coord is higher than MCP
            tip_y = lm_list[i][1]
            pip_y = lm_list[i-2][1]
            mcp_y = lm_list[i-3][1]
            # In standard upright orientation, check if tip is in-between PIP and MCP
            # If not in upright, we check distances
            d_tip_mcp = self.get_distance(lm_list[i], lm_list[i-3])
            d_pip_mcp = self.get_distance(lm_list[i-2], lm_list[i-3])
            if d_tip_mcp > d_pip_mcp * 1.5 or d_tip_mcp < d_pip_mcp * 0.5:
                is_curved = False
                break
        
        if is_curved and avg_tip_dist > palm_size * 0.9 and avg_tip_dist < palm_size * 1.4:
            return 'C', 0.85

        # Default fallback: check if we are closer to B or A
        open_count = sum([index_open, middle_open, ring_open, pinky_open])
        if open_count >= 3:
            return 'B', 0.6
        else:
            return 'A', 0.5

    def process_frame(self, frame, mode="heuristic"):
        """
        Process a single image frame (BGR format).
        Returns:
            - prediction: string (e.g. 'A', 'B', 'SPACE', etc.)
            - confidence: float (0.0 to 1.0)
            - landmarks: list of 21 normalized [x, y] coordinates (or None if no hand)
            - raw_landmarks: list of 21 raw [x, y, z] coordinates
        """
        if self.hands is None:
            return None, 0.0, None, None
            
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if not results.multi_hand_landmarks:
            return None, 0.0, None, None
            
        # We only look at the first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Get frame dimensions
        h, w, _ = frame.shape
        
        # Extract coordinates
        lm_list = []
        drawing_lms = [] # For drawing on canvas: pixel values
        for lm in hand_landmarks.landmark:
            lm_list.append([lm.x, lm.y, lm.z])
            drawing_lms.append({"x": lm.x, "y": lm.y})
            
        # Predict based on mode
        prediction, confidence = self.process_landmarks(lm_list, mode)
            
        return prediction, confidence, drawing_lms, lm_list

    def process_landmarks(self, lm_list, mode="heuristic"):
        """
        Process a list of 21 raw landmarks [[x, y, z], ...] directly.
        Returns:
            - prediction: string (e.g. 'A', 'B', 'SPACE', etc.)
            - confidence: float (0.0 to 1.0)
        """
        prediction = None
        confidence = 0.0
        
        if mode == "ml" and self.model is not None:
            try:
                normalized = self.normalize_landmarks(lm_list)
                prediction_idx = self.model.predict([normalized])[0]
                prediction = self.labels_map.get(prediction_idx, 'UNKNOWN')
                
                # Get prediction probabilities if available
                probabilities = self.model.predict_proba([normalized])[0]
                confidence = float(np.max(probabilities))
            except Exception as e:
                print(f"ML Prediction failed: {e}")
                mode = "heuristic" # Fallback
                
        if mode == "heuristic" or prediction is None:
            prediction, confidence = self.classify_heuristic(lm_list)
            
        return prediction, confidence


    def draw_landmarks(self, frame, raw_landmarks):
        # Helper to draw landmarks on an OpenCV frame (useful for debugging/local testing)
        # raw_landmarks should be a list of 21 [x, y, z] coordinates
        if not raw_landmarks:
            return frame
            
        # Recreate MediaPipe landmark list to use mp_draw
        from mediapipe.framework.formats import landmark_pb2
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        for lm in raw_landmarks:
            l = hand_landmarks_proto.landmark.add()
            l.x = lm[0]
            l.y = lm[1]
            l.z = lm[2]
            
        self.mp_draw.draw_landmarks(
            frame,
            hand_landmarks_proto,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_drawing_styles.get_default_hand_landmarks_style(),
            self.mp_drawing_styles.get_default_hand_connections_style()
        )
        return frame
