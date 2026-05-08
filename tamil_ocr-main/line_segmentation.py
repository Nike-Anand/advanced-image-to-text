import cv2
import numpy as np

def segment_lines(image_path):
    """Simple line segmentation using horizontal projection"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Threshold to binary
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Horizontal projection
    horizontal_projection = np.sum(binary, axis=1)
    
    # Find line boundaries
    lines = []
    in_line = False
    start = 0
    
    for i, val in enumerate(horizontal_projection):
        if val > 0 and not in_line:  # Start of line
            start = i
            in_line = True
        elif val == 0 and in_line:  # End of line
            if i - start > 10:  # Minimum line height
                lines.append((start, i))
            in_line = False
    
    # Extract line images
    line_images = []
    for start, end in lines:
        line_img = img[start:end, :]
        line_images.append(line_img)
    
    return line_images

if __name__ == "__main__":
    lines = segment_lines("test_images/0.jpg")
    print(f"Found {len(lines)} lines")
    
    # Save line images for testing
    for i, line in enumerate(lines):
        cv2.imwrite(f"line_{i}.jpg", line)