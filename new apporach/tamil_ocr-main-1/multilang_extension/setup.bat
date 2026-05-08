@echo off
echo Setting up PARSEQ multi-language extension...

REM Create necessary directories
mkdir models 2>nul
mkdir datasets\multilang_train 2>nul
mkdir datasets\multilang_val 2>nul
mkdir fonts 2>nul

echo Directories created.

REM Check if PARSEQ exists
if not exist "parseq" (
    echo Cloning PARSEQ repository...
    git clone https://github.com/baudm/parseq.git
    cd parseq
    pip install -r requirements.txt
    cd ..
)

REM Download fonts for different languages
echo Downloading language fonts...

if not exist "fonts\NotoSansDevanagari-Regular.ttf" (
    curl -L "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansDevanagari/NotoSansDevanagari-Regular.ttf" -o "fonts\NotoSansDevanagari-Regular.ttf"
)

if not exist "fonts\NotoSansTelugu-Regular.ttf" (
    curl -L "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansTelugu/NotoSansTelugu-Regular.ttf" -o "fonts\NotoSansTelugu-Regular.ttf"
)

if not exist "fonts\NotoSansKannada-Regular.ttf" (
    curl -L "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansKannada/NotoSansKannada-Regular.ttf" -o "fonts\NotoSansKannada-Regular.ttf"
)

if not exist "fonts\NotoSans-Regular.ttf" (
    curl -L "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSans/NotoSans-Regular.ttf" -o "fonts\NotoSans-Regular.ttf"
)

echo Fonts downloaded.

REM Install additional requirements
echo Installing Python requirements...
pip install pillow torch torchvision pytorch-lightning

echo Setup complete!
echo.
echo Next steps:
echo 1. Generate synthetic data: python data_generator.py
echo 2. Train multi-language model: python train_multilang.py
echo 3. Use multi-language OCR: python multilang_ocr.py

pause