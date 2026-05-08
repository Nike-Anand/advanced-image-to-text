# Essential Tamil OCR

This folder contains only the essential files needed to run Tamil OCR.

## Setup
```bash
pip install -r requirements_minimal.txt
```

## Usage
```python
python run_ocr.py
```

## Files Structure
- `ocr_tamil/` - Core OCR module
- `test_images/` - Sample test images  
- `run_ocr.py` - Simple test script
- `requirements_minimal.txt` - Minimal dependencies

## Features
- Text detection + recognition
- Offline functionality
- Tamil + English support
- GPU acceleration (if available)