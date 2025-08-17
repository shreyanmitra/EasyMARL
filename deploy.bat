@echo off
REM EasyMARL Deployment Script for Windows
REM This script sets up the environment and installs all required dependencies

echo === EasyMARL Deployment Script ===
echo Setting up Multi-Agent Reinforcement Learning Environment
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed. Please install Python 3.8+ first.
    pause
    exit /b 1
)

REM Display Python version
echo Python version:
python --version

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo Installing requirements from requirements.txt...
pip install -r requirements.txt

REM Install additional MARL dependencies
echo Installing additional MARL dependencies...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install wandb
pip install matplotlib seaborn
pip install moviepy
pip install gym==0.21.0
pip install numpy==1.21.0
pip install pyyaml
pip install pillow

REM Install development dependencies
echo Installing development dependencies...
pip install pytest
pip install black
pip install flake8
pip install mypy

echo.
echo === Installation Complete ===
echo To activate the environment in the future, run:
echo venv\Scripts\activate.bat
echo.
echo To test the installation, run:
echo python test_framework.py
echo.
echo To start training, run:
echo python main.py --algorithm ippo --env_name MultiGrid-Cluttered-Fixed-15x15
pause
