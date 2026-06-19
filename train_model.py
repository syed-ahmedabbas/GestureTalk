import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def main():
    data_dir = "data"
    gestures = ['A', 'B', 'C', 'L', 'Y', 'SPACE', 'DELETE', 'CLEAR']
    
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' not found. Please run collect_data.py first.")
        return
        
    X = []
    y = []
    
    # Load all collected data
    for class_folder in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_folder)
        if not os.path.isdir(class_path):
            continue
            
        try:
            class_idx = int(class_folder)
            if class_idx < 0 or class_idx >= len(gestures):
                print(f"Skipping folder '{class_folder}': out of range.")
                continue
        except ValueError:
            print(f"Skipping folder '{class_folder}': not a valid gesture index.")
            continue
            
        files = [f for f in os.listdir(class_path) if f.endswith('.npy')]
        print(f"Loading {len(files)} samples for gesture '{gestures[class_idx]}' (index {class_idx})")
        
        for file in files:
            file_path = os.path.join(class_path, file)
            try:
                data = np.load(file_path)
                # Ensure data is 63 features (21 landmarks * 3 coords)
                if len(data) == 63:
                    X.append(data)
                    y.append(class_idx)
                else:
                    print(f"Warning: Sample {file} has invalid dimension {len(data)}. Expected 63. Skipping.")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                
    if len(X) == 0:
        print("No valid training samples found. Please collect data first.")
        return
        
    X = np.array(X)
    y = np.array(y)
    
    # Check class distribution
    unique_classes, counts = np.unique(y, return_counts=True)
    print("\nClass distribution in loaded dataset:")
    for c, count in zip(unique_classes, counts):
        print(f"  Gesture '{gestures[c]}': {count} samples")
        
    if len(unique_classes) < 2:
        print("\nWarning: You need at least 2 distinct gestures in your dataset to train a classifier.")
        print("Falling back to training on a mixture of synthetic and collected data is recommended.")
        
    # Split training and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if len(unique_classes) >= 2 else None)
    
    print(f"\nTraining set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")
    
    # Train Random Forest
    print("\nTraining Random Forest model...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel accuracy on test set: {accuracy * 100:.2f}%")
    
    # Print detailed report
    target_names = [gestures[c] for c in unique_classes]
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # Save the model
    os.makedirs("model", exist_ok=True)
    model_path = "model/model.p"
    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)
        
    print(f"Trained model saved successfully to {model_path}!")

if __name__ == "__main__":
    main()
