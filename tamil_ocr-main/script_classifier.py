import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

class ScriptClassifier:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=50, random_state=42)
        self.is_trained = False
    
    def extract_features(self, image):
        """Extract simple features for script classification"""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Resize to standard size
        image = cv2.resize(image, (128, 32))
        
        # Basic features
        features = []
        
        # 1. Pixel density in different regions
        h, w = image.shape
        regions = [
            image[:h//3, :],      # Top
            image[h//3:2*h//3, :], # Middle  
            image[2*h//3:, :],    # Bottom
        ]
        
        for region in regions:
            density = np.sum(region < 128) / region.size
            features.append(density)
        
        # 2. Horizontal line density (Tamil has more horizontal strokes)
        kernel = np.ones((1, 5), np.uint8)
        horizontal = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        h_density = np.sum(horizontal < 128) / horizontal.size
        features.append(h_density)
        
        # 3. Vertical line density
        kernel = np.ones((5, 1), np.uint8)
        vertical = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        v_density = np.sum(vertical < 128) / vertical.size
        features.append(v_density)
        
        # 4. Aspect ratio of connected components
        contours, _ = cv2.findContours(255 - image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        aspect_ratios = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 5 and h > 5:  # Filter small noise
                aspect_ratios.append(w / h)
        
        if aspect_ratios:
            features.extend([np.mean(aspect_ratios), np.std(aspect_ratios)])
        else:
            features.extend([1.0, 0.0])
        
        return np.array(features)
    
    def train_simple_model(self):
        """Train with basic heuristics (placeholder for real training)"""
        # This is a simplified version - in practice, you'd train on real data
        # For now, we'll use feature-based rules
        self.is_trained = True
        print("Script classifier ready (rule-based)")
    
    def predict_script(self, image):
        """Predict script: TA (Tamil), EN (English), or UNKNOWN"""
        features = self.extract_features(image)
        
        # Simple rule-based classification (replace with trained model)
        h_density = features[3]  # Horizontal line density
        v_density = features[4]  # Vertical line density
        
        # Tamil typically has more horizontal strokes
        if h_density > 0.15 and h_density > v_density * 1.5:
            return "TA"
        elif v_density > 0.1 and features[5] > 2.0:  # English has more varied aspect ratios
            return "EN"
        else:
            return "UNKNOWN"

# Test the classifier
if __name__ == "__main__":
    classifier = ScriptClassifier()
    classifier.train_simple_model()
    
    # Test with sample image
    img = cv2.imread("test_images/0.jpg")
    if img is not None:
        script = classifier.predict_script(img)
        print(f"Detected script: {script}")
    else:
        print("Test image not found")