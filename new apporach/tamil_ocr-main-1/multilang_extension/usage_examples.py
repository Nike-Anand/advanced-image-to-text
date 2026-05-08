"""
Multi-Language OCR Usage Examples
Demonstrates how to use the extended Tamil OCR with additional languages
"""

import sys
import os
sys.path.append('..')

from multilang_ocr import MultiLangOCR, create_multilang_ocr

def demo_single_language():
    """Demo with single language (Tamil + English)"""
    print("=== Single Language Demo (Tamil + English) ===")
    
    ocr = MultiLangOCR(detect=True, lang=["tamil", "english"])
    
    # Example usage (replace with actual image path)
    # results = ocr.predict("test_images/tamil_text.jpg")
    # print("Results:", results)
    
    print("OCR initialized for Tamil + English")

def demo_multi_language():
    """Demo with multiple languages"""
    print("\n=== Multi-Language Demo (Tamil + English + Hindi) ===")
    
    ocr = create_multilang_ocr(["tamil", "english", "hindi"])
    
    # Example batch processing
    # image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
    # results = ocr.predict(image_paths)
    # 
    # for i, result in enumerate(results):
    #     print(f"Image {i+1}: {result}")
    
    print("OCR initialized for Tamil + English + Hindi")

def demo_all_languages():
    """Demo with all supported languages"""
    print("\n=== All Languages Demo ===")
    
    languages = ["tamil", "english", "hindi", "telugu", "kannada"]
    ocr = create_multilang_ocr(languages)
    
    print(f"OCR initialized for: {', '.join(languages)}")

def demo_configuration_options():
    """Demo various configuration options"""
    print("\n=== Configuration Options Demo ===")
    
    # High accuracy configuration
    ocr_high_acc = MultiLangOCR(
        detect=True,
        lang=["tamil", "english", "hindi"],
        batch_size=8,  # Smaller batch for better accuracy
        recognize_thres=0.90,  # Higher threshold
        text_threshold=0.7,  # More strict text detection
        details=2  # Include bounding boxes and confidence
    )
    
    # Fast processing configuration
    ocr_fast = MultiLangOCR(
        detect=True,
        lang=["tamil", "english"],
        batch_size=32,  # Larger batch for speed
        recognize_thres=0.80,  # Lower threshold
        assume_straight_page=True,  # Faster processing
        fp16=True  # Half precision for speed
    )
    
    print("High accuracy OCR configured")
    print("Fast processing OCR configured")

def demo_line_by_line_processing():
    """Demo line-by-line text extraction"""
    print("\n=== Line-by-Line Processing Demo ===")
    
    def line_print(prediction):
        """Format output by lines"""
        current_line = 1
        extracted_text = ""
        for text in prediction:
            pred_text = text[0]
            line_details = text[2][1] if len(text) > 2 else 1
            
            if line_details != current_line:
                extracted_text += "\n" + pred_text + " "
                current_line = line_details
            else:
                extracted_text += pred_text + " "
        return extracted_text
    
    ocr = MultiLangOCR(detect=True, details=2, batch_size=128, lang=["tamil", "english", "hindi"])
    
    # Example usage:
    # image_path = "multi_line_text.jpg"
    # text_list = ocr.predict([image_path])
    # formatted_text = line_print(text_list[0])
    # print("Formatted text:")
    # print(formatted_text)
    
    print("Line-by-line processing configured")

def performance_comparison():
    """Compare performance with different configurations"""
    print("\n=== Performance Comparison ===")
    
    configs = [
        {"name": "Tamil Only", "lang": ["tamil"]},
        {"name": "Tamil + English", "lang": ["tamil", "english"]},
        {"name": "Tamil + English + Hindi", "lang": ["tamil", "english", "hindi"]},
        {"name": "All Languages", "lang": ["tamil", "english", "hindi", "telugu", "kannada"]}
    ]
    
    for config in configs:
        ocr = MultiLangOCR(lang=config["lang"])
        charset_size = len(ocr.multilang_charset) if hasattr(ocr, 'multilang_charset') else 0
        print(f"{config['name']}: {len(config['lang'])} languages, ~{charset_size} characters")

def main():
    """Run all demos"""
    print("Multi-Language Tamil OCR Extension Demo")
    print("=" * 50)
    
    try:
        demo_single_language()
        demo_multi_language()
        demo_all_languages()
        demo_configuration_options()
        demo_line_by_line_processing()
        performance_comparison()
        
        print("\n" + "=" * 50)
        print("All demos completed successfully!")
        print("\nTo use with actual images:")
        print("1. Replace image paths in the demo functions")
        print("2. Ensure you have trained multi-language models")
        print("3. Run: python usage_examples.py")
        
    except Exception as e:
        print(f"Demo error: {e}")
        print("Make sure all dependencies are installed and models are available")

if __name__ == "__main__":
    main()