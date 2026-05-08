#!/bin/bash

# Setup script for PARSEQ multi-language extension

echo "Setting up PARSEQ multi-language extension..."

# Create necessary directories
mkdir -p models
mkdir -p datasets/multilang_train
mkdir -p datasets/multilang_val
mkdir -p fonts

echo "Directories created."

# Download PARSEQ repository (if not exists)
if [ ! -d "parseq" ]; then
    echo "Cloning PARSEQ repository..."
    git clone https://github.com/baudm/parseq.git
    cd parseq
    pip install -r requirements.txt
    cd ..
fi

# Download fonts for different languages
echo "Downloading language fonts..."

# Hindi font
if [ ! -f "fonts/NotoSansDevanagari-Regular.ttf" ]; then
    curl -L "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansDevanagari/NotoSansDevanagari-Regular.ttf" -o "fonts/NotoSansDevanagari-Regular.ttf"
fi

# Telugu font
if [ ! -f "fonts/NotoSansTelugu-Regular.ttf" ]; then
    curl -L "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansTelugu/NotoSansTelugu-Regular.ttf" -o "fonts/NotoSansTelugu-Regular.ttf"
fi

# Kannada font
if [ ! -f "fonts/NotoSansKannada-Regular.ttf" ]; then
    curl -L "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansKannada/NotoSansKannada-Regular.ttf" -o "fonts/NotoSansKannada-Regular.ttf"
fi

# Default font
if [ ! -f "fonts/NotoSans-Regular.ttf" ]; then
    curl -L "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSans/NotoSans-Regular.ttf" -o "fonts/NotoSans-Regular.ttf"
fi

echo "Fonts downloaded."

# Install additional requirements
echo "Installing Python requirements..."
pip install pillow torch torchvision pytorch-lightning

echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Generate synthetic data: python data_generator.py"
echo "2. Train multi-language model: python train_multilang.py"
echo "3. Use multi-language OCR: python multilang_ocr.py"