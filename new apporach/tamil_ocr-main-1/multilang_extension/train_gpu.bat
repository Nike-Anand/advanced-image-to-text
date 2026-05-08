@echo off
echo ========================================
echo Multi-Language Tamil OCR - GPU Training
echo ========================================

echo.
echo Step 1: Checking GPU...
python -c "import torch; print('GPU Available:', torch.cuda.is_available()); print('GPU Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

echo.
echo Step 2: Quick Setup Check...
python quick_start.py

echo.
echo Step 3: Generate Training Data...
echo Running: python data_generator.py
python data_generator.py

echo.
echo Step 4: Start GPU-Forced Training...
echo Running: python train_multilang.py
python train_multilang.py

echo.
echo Step 5: Test Multi-Language OCR...
echo Running: python usage_examples.py
python usage_examples.py

echo.
echo ========================================
echo Training Complete!
echo ========================================
pause