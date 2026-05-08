from google import genai
from google.genai import types
import os
import time

# Initialize the client with your API key
client = genai.Client(api_key="AIzaSyD3hAaDltsYhYjJAYBiXonSAyMAqdk9XyE")

def ocr_handwritten_text(image_path):
    """Extract handwritten text from image using Gemini API"""
    start_time = time.time()
    try:
        # Read image as bytes
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        
        # Call Gemini API
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type='image/jpeg',
                ),
                ' Make the temperature at 0.Transcribe the handwritten text in this image exactly.'
            ]
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        print(f"Processing time: {processing_time:.2f} seconds")
        
        return response.text.strip()
    
    except Exception as e:
        end_time = time.time()
        processing_time = end_time - start_time
        print(f"Processing time: {processing_time:.2f} seconds")
        return f"Error: {str(e)}"

if __name__ == "__main__":
    # Example usage
    image_path = "C:\\Users\\nikea\\Desktop\\test\\test.jpeg"  # Replace with your image path
    
    if os.path.exists(image_path):
        extracted_text = ocr_handwritten_text(image_path)
        print("Extracted Text:")
        print(extracted_text)
    else:
        print(f"Image file '{image_path}' not found.")