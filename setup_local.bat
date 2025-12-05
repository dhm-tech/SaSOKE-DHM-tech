@echo off
echo ============================================
echo SaSOKE Local Setup - Windows
echo ============================================

REM Create required directories
echo Creating directories...
mkdir deps 2>nul
mkdir deps\smpl-x 2>nul
mkdir deps\mbart-h2s-csl-phoenix 2>nul
mkdir deps\t2m 2>nul
mkdir checkpoints 2>nul
mkdir checkpoints\vae 2>nul
mkdir data 2>nul

echo.
echo ============================================
echo Directory structure created!
echo ============================================
echo.
echo Now you need to:
echo.
echo 1. COPY TOKENIZER from Yousef:
echo    - Put tokenizer.ckpt in: checkpoints\vae\tokenizer.ckpt
echo.
echo 2. DOWNLOAD DEPENDENCIES (from Google Drive):
echo    - smpl-x/mean.pt and std.pt
echo    - deps/mbart-h2s-csl-phoenix/ folder
echo    - deps/t2m/ folder
echo.
echo 3. ORGANIZE DATA:
echo    - Your pose files (00_OSX_params, etc.) should be linked to data/
echo.
echo Run 'pip install -r requirements.txt' to install Python packages
echo.
pause

