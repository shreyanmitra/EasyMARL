@echo off
REM EasyMARL Quick Start Script for Windows

echo 🎓 EasyMARL Quick Start for Windows
echo =====================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python not found. Please install Python 3.8+ from python.org
    pause
    exit /b 1
)

echo ✅ Python found
echo.

REM Check if pip is available
python -m pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ pip not found. Please install pip
    pause
    exit /b 1
)

echo ✅ pip found
echo.

REM Install/check EasyMARL
echo 📦 Checking EasyMARL installation...
python -c "import easymarl" >nul 2>&1
if %errorlevel% neq 0 (
    echo 📦 Installing EasyMARL...
    python -m pip install easymarl
    if %errorlevel% neq 0 (
        echo ❌ Failed to install EasyMARL
        pause
        exit /b 1
    )
    echo ✅ EasyMARL installed successfully!
) else (
    echo ✅ EasyMARL already installed
)

echo.
echo 🚀 Launching EasyMARL...
echo.

REM Try to launch the GUI
python -c "import easymarl; easymarl.launch_gui()" 2>nul
if %errorlevel% neq 0 (
    echo ❌ Could not launch GUI interface
    echo 📖 Alternative: Run 'python quick-start.py' for more options
    pause
    exit /b 1
)

echo ✅ EasyMARL launched successfully!
pause
