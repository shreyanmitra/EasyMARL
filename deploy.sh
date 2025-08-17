#!/bin/bash

# EasyMARL Deployment Script
# This script sets up the environment and installs all required dependencies

echo "=== EasyMARL Deployment Script ==="
echo "Setting up Multi-Agent Reinforcement Learning Environment"
echo ""

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "ERROR: Python is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check Python version
python_version=$(python --version 2>&1 | cut -d' ' -f2)
echo "Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements from requirements.txt..."
pip install -r requirements.txt

# Install additional MARL dependencies
echo "Installing additional MARL dependencies..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install wandb
pip install matplotlib seaborn
pip install moviepy
pip install gym==0.21.0
pip install numpy==1.21.0
pip install pyyaml
pip install pillow

# Install development dependencies
echo "Installing development dependencies..."
pip install pytest
pip install black
pip install flake8
pip install mypy

echo ""
echo "=== Installation Complete ==="
echo "To activate the environment in the future, run:"
echo "source venv/bin/activate  (on Linux/Mac)"
echo "venv\\Scripts\\activate    (on Windows)"
echo ""
echo "To test the installation, run:"
echo "python test_framework.py"
echo ""
echo "To start training, run:"
echo "python main.py --algorithm ippo --env_name MultiGrid-Cluttered-Fixed-15x15"
