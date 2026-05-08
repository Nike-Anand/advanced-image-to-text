import sys
import os

# Set UTF-8 encoding for Windows console
if sys.platform == "win32":
    os.system("chcp 65001 > nul")
    sys.stdout.reconfigure(encoding='utf-8')

# Test offline capability by checking if models exist locally
from pathlib import Path
import requests

def test_offline_capability():
    model_save_location = os.path.join(Path.home(), ".model_weights")
    tamil_model = os.path.join(model_save_location, "parseq_tamil_v3.pt")
    detect_model = os.path.join(model_save_location, "craft_mlt_25k.pth")
    
    print("Checking model files:")
    print(f"Tamil model exists: {os.path.exists(tamil_model)}")
    print(f"Detection model exists: {os.path.exists(detect_model)}")
    
    if os.path.exists(tamil_model) and os.path.exists(detect_model):
        print("✅ Models are available locally - OCR can work OFFLINE")
        return True
    else:
        print("❌ Models need to be downloaded first - requires internet connection")
        return False

def test_network_dependency():
    # Temporarily patch requests to simulate offline
    original_get = requests.get
    
    def mock_get(*args, **kwargs):
        raise requests.exceptions.ConnectionError("Simulated offline mode")
    
    try:
        # Test if OCR works without network
        requests.get = mock_get
        
        from ocr_tamil.ocr import OCR
        print("✅ OCR module loaded successfully in offline mode")
        
        # Test prediction
        ocr = OCR(detect=True, details=2, text_threshold=0.3, fp16=False)
        image_path = r"test_images\0.jpg"
        text_list = ocr.predict(image_path)
        print("✅ OCR prediction works offline!")
        print(f"Result: {text_list[0][0][0]} (confidence: {text_list[0][0][1]:.3f})")
        
    except Exception as e:
        print(f"❌ Offline test failed: {e}")
    finally:
        # Restore original requests
        requests.get = original_get

if __name__ == "__main__":
    print("=== Tamil OCR Offline Capability Test ===\n")
    
    # Check if models exist locally
    models_available = test_offline_capability()
    
    if models_available:
        print("\n=== Testing Network Independence ===")
        test_network_dependency()
    else:
        print("\n⚠️  Run the OCR once with internet to download models, then it will work offline")