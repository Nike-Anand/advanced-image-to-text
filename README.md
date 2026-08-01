
#  Advanced Tamil Handwritten Text Recognition (HWR / OCR)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Framework](https://img.shields.io/badge/PyTorch%2FTensorFlow-Deep%20Learning-orange.svg)
![Domain](https://img.shields.io/badge/Domain-OCR%20%7C%20Computer%20Vision-green.svg)
![Language](https://img.shields.io/badge/Language-Tamil%20(%E0%AE%A4%E0%AE%AE%E0%AE%BF%E0%AE%B4%E0%AF%8D)-red.svg)

An end-to-end deep learning framework designed to accurately extract and transcribe handwritten Tamil text from images into digital Unicode text. Tamil handwritten recognition presents unique challenges due to complex character shapes, connected strokes, and diverse writing styles—this project solves them using state-of-the-art vision and sequential models.

---

##  Key Features

* **Advanced Image Preprocessing:** Adaptive thresholding, binarization, noise reduction, and skew correction optimized for Tamil scripts.
* **Character & Line Segmentation:** Automated line, word, and character extraction from unconstrained document images.
* **Deep Learning Engine:** Powered by a **CRNN (CNN + BiLSTM + CTC Loss)** / **Vision Transformer (ViT)** architecture to handle complex sequential dependencies in Tamil handwriting.
* **Unicode Generation:** Converts model predictions seamlessly into standardized Tamil Unicode text.
* **Batch Processing & Inference API:** Easily process single image files or entire document directories.

---

##  Architecture & Pipeline

```text
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│  Input Image    │ ──► │ Preprocessing    │ ──► │  Segmentation /     │
│  (Tamil Text)   │     │ (Denoise & Skew) │     │  Feature Extraction │
└─────────────────┘     └──────────────────┘     └─────────────────────┘
                                                            │
┌─────────────────┐     ┌──────────────────┐                ▼
│ Output Tamil    │ ◄── │  CTC Decoding /  │ ◄── ┌─────────────────────┐
│ Unicode Text    │     │  Language Model  │     │ Deep Neural Network │
└─────────────────┘     └──────────────────┘     │ (CNN + BiLSTM)      │
                                                 └─────────────────────┘

```

1. **Preprocessing:** Noise filtering, contrast adjustment, and resizing.
2. **Feature Extraction:** Convolutional Neural Network (CNN) extracts high-level spatial visual features.
3. **Sequence Modeling:** Bidirectional LSTM (BiLSTM) captures context across characters.
4. **Transcription:** CTC Loss / Transformer Decoder yields final character predictions.

---

##  Getting Started

### 1. Prerequisites

Ensure you have **Python 3.8+** and a GPU runtime (recommended for training/inference).

### 2. Installation

```bash
# Clone the repository
git clone [https://github.com/Nike-Anand/advanced-image-to-text.git](https://github.com/Nike-Anand/advanced-image-to-text.git)
cd advanced-image-to-text

# Install dependencies
pip install -r requirements.txt

```

---

##  Usage

### Quickstart Example

```python
from model import TamilOCR

# Initialize OCR Pipeline
ocr = TamilOCR(weights_path="weights/best_tamil_hwr.pth")

# Run inference on an image
image_path = "sample_images/tamil_handwritten_note.jpg"
transcribed_text = ocr.predict(image_path)

print("Recognized Text:")
print(transcribed_text)

```

---

##  Datasets & Performance

* **Datasets Used:** HP LABS Tamil Handwritten Character Dataset, custom augmented datasets, and synthetic handwritten samples.
* **Evaluation Metrics:**
* **Character Error Rate (CER):** Target $< 5\%$
* **Word Error Rate (WER):** Target $< 12\%$



---

##  Roadmap

* [ ] Add support for mixed language (Tamil + English) document recognition.
* [ ] Integrate a web dashboard / UI using Streamlit or Gradio.
* [ ] Optimize model with ONNX Runtime for edge and mobile deployment.
* [ ] Add language model post-processing (Grammar/Spell Check for Tamil).

---

##  Contributing

Contributions are always welcome! Feel free to open an issue or submit a pull request if you'd like to improve the recognition accuracy, dataset augmentation, or UI.

---


```

```
