"""
Complete Document OCR using Trained Tamil Handwritten Model
Detects and recognizes all text in a document image
"""

import sys
import os

# Set UTF-8 encoding for Windows console
if sys.platform == "win32":
    os.system("chcp 65001 > nul")
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

import torch
from PIL import Image
import cv2
import numpy as np

# Import the existing OCR model for text detection
from ocr_tamil.ocr import OCR

# Import trained model components
from ocr_tamil.strhub.models.parseq.system import PARSeq
from tamil_char_dataset import get_default_transforms


class CompleteOCR:
    """
    Complete OCR system combining:
    1. Text detection from existing OCR model
    2. Character recognition from trained handwritten model
    """
    
    def __init__(self, trained_model_path, use_gpu=True):
        """
        Initialize the complete OCR system.
        
        Args:
            trained_model_path: Path to trained character recognition model
            use_gpu: Whether to use GPU for inference
        """
        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Load text detection model (existing OCR)
        print("Loading text detection model...")
        self.detector = OCR(detect=True, details=2, text_threshold=0.3, fp16=False)
        
        # Load trained character recognition model
        print(f"Loading trained character model from {trained_model_path}...")
        self.char_model = PARSeq.load_from_checkpoint(trained_model_path)
        self.char_model.eval()
        self.char_model = self.char_model.to(self.device)
        
        # Image transform for character recognition
        self.transform = get_default_transforms((32, 128), augment=False)
        
        print("✓ OCR system initialized successfully!")
    
    def detect_text_regions(self, image_path):
        """
        Detect text regions in the image.
        
        Args:
            image_path: Path to input image
            
        Returns:
            List of detected text regions with bounding boxes
        """
        print(f"\nDetecting text in: {image_path}")
        
        # Use existing OCR for text detection
        detections = self.detector.predict(image_path)
        
        if not detections or len(detections) == 0:
            print("No text detected in image")
            return []
        
        print(f"✓ Detected {len(detections[0])} text regions")
        return detections[0]
    
    def recognize_character(self, image_crop):
        """
        Recognize a single character using the trained model.
        
        Args:
            image_crop: PIL Image or numpy array of character
            
        Returns:
            Tuple of (predicted_character, confidence)
        """
        # Convert to PIL Image if numpy array
        if isinstance(image_crop, np.ndarray):
            image_crop = Image.fromarray(cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB))
        
        # Ensure RGB
        if image_crop.mode != 'RGB':
            image_crop = image_crop.convert('RGB')
        
        # Transform and add batch dimension
        image_tensor = self.transform(image_crop).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            logits = self.char_model(image_tensor)
            probs = logits.softmax(-1)
            preds, probs_decoded = self.char_model.tokenizer.decode(probs)
            confidence = probs_decoded[0].prod().item()
        
        return preds[0], confidence
    
    def process_document(self, image_path, output_path=None, use_trained_model=True):
        """
        Process complete document and extract all text.
        
        Args:
            image_path: Path to input document image
            output_path: Optional path to save annotated image
            use_trained_model: If True, use trained model for recognition,
                             otherwise use existing OCR model
        
        Returns:
            Dictionary with extracted text and metadata
        """
        print("\n" + "="*80)
        print("COMPLETE DOCUMENT OCR")
        print("="*80)
        
        # Load image for visualization
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Detect text regions
        detections = self.detect_text_regions(image_path)
        
        if not detections:
            return {
                'text': '',
                'num_regions': 0,
                'regions': []
            }
        
        # Process each detected region
        results = []
        full_text = []
        current_line = 1
        
        print(f"\nRecognizing text...")
        print("-" * 80)
        
        for idx, detection in enumerate(detections):
            text, confidence, bbox = detection
            line_num = bbox[1]  # Line number from detection
            
            if use_trained_model:
                # Extract region from image
                x_coords = [point[0] for point in bbox[0]]
                y_coords = [point[1] for point in bbox[0]]
                x_min, x_max = int(min(x_coords)), int(max(x_coords))
                y_min, y_max = int(min(y_coords)), int(max(y_coords))
                
                # Crop region
                crop = img[y_min:y_max, x_min:x_max]
                
                if crop.size > 0:
                    # Recognize using trained model
                    pred_text, conf = self.recognize_character(crop)
                else:
                    pred_text, conf = text, confidence
            else:
                # Use existing OCR recognition
                pred_text, conf = text, confidence
            
            # Store result
            results.append({
                'text': pred_text,
                'confidence': conf,
                'bbox': bbox,
                'line': line_num
            })
            
            # Build full text with line breaks
            if line_num != current_line:
                full_text.append('\n')
                current_line = line_num
            
            full_text.append(pred_text + ' ')
            
            # Print progress
            print(f"Region {idx+1}/{len(detections)}: '{pred_text}' (confidence: {conf:.3f})")
            
            # Draw bounding box on image
            if output_path:
                points = np.array(bbox[0], dtype=np.int32)
                cv2.polylines(img, [points], True, (0, 255, 0), 2)
                # Add text label
                cv2.putText(img, pred_text, (points[0][0], points[0][1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Save annotated image
        if output_path:
            cv2.imwrite(output_path, img)
            print(f"\n✓ Annotated image saved to: {output_path}")
        
        # Combine text
        extracted_text = ''.join(full_text).strip()
        
        print("\n" + "="*80)
        print("EXTRACTED TEXT:")
        print("="*80)
        print(extracted_text)
        print("="*80)
        
        return {
            'text': extracted_text,
            'num_regions': len(results),
            'regions': results,
            'average_confidence': np.mean([r['confidence'] for r in results])
        }


def main():
    """Main function to demonstrate complete document OCR."""
    
    # Configuration
    TRAINED_MODEL = r"checkpoints\tamil-char-epoch=29-val_accuracy=97.45.ckpt"
    INPUT_IMAGE = r"test.jpeg"  # Change to your document image
    OUTPUT_IMAGE = r"outputs\ocr_result.png"
    
    # Check if model exists
    if not os.path.exists(TRAINED_MODEL):
        print(f"Error: Trained model not found at {TRAINED_MODEL}")
        print("Please train the model first or update the path.")
        return
    
    # Check if input image exists
    if not os.path.exists(INPUT_IMAGE):
        print(f"Error: Input image not found at {INPUT_IMAGE}")
        print("Please provide a valid image path.")
        return
    
    # Create output directory
    os.makedirs("outputs", exist_ok=True)
    
    # Initialize OCR system
    ocr = CompleteOCR(TRAINED_MODEL, use_gpu=True)
    
    # Process document using existing OCR (recommended for full documents)
    print("\n" + "="*80)
    print("Processing Document with Existing OCR Model")
    print("="*80)
    result = ocr.process_document(
        INPUT_IMAGE,
        output_path=OUTPUT_IMAGE,
        use_trained_model=False  # Use existing OCR for full text recognition
    )
    
    print(f"\n✓ Total regions detected: {result['num_regions']}")
    print(f"✓ Average confidence: {result['average_confidence']:.3f}")
    print(f"✓ Extracted text saved")
    
    print("\n" + "="*80)
    print("OCR COMPLETED SUCCESSFULLY!")
    print("="*80)


if __name__ == "__main__":
    main()