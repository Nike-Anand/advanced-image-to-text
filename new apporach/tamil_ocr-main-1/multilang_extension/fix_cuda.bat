@echo off
echo ========================================
echo CUDA PyTorch Installation Fix
echo RTX 4060 Detected - Installing CUDA Support
echo ========================================

echo.
echo Current PyTorch version:
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA Available:', torch.cuda.is_available())"

echo.
echo Installing PyTorch with CUDA 11.8 support...
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo.
echo Verifying CUDA installation...
python -c "import torch; print('✅ PyTorch:', torch.__version__); print('✅ CUDA Available:', torch.cuda.is_available()); print('✅ GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

echo.
echo Testing GPU training readiness...
python quick_start.py

pause