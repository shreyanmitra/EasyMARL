#!/bin/bash

# EasyMARL Combined Deployment Script
# This script handles both Python backend and React frontend deployment

echo "🚀 EasyMARL Full Stack Deployment"
echo "=================================="
echo ""

# Check if we're in the right directory
if [ ! -f "requirements.txt" ] || [ ! -f "react-frontend/package.json" ]; then
    echo "❌ Error: Please run this script from the EasyMARL root directory"
    echo "   Make sure both requirements.txt and react-frontend/package.json exist"
    exit 1
fi

# =============================================================================
# PHASE 1: Python Backend Setup
# =============================================================================

echo "📦 PHASE 1: Setting up Python Backend"
echo "======================================"

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "❌ ERROR: Python is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check Python version
python_version=$(python --version 2>&1 | cut -d' ' -f2)
echo "✅ Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📋 Creating virtual environment..."
    python -m venv venv
    if [ $? -ne 0 ]; then
        echo "❌ Error: Failed to create virtual environment"
        exit 1
    fi
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing requirements from requirements.txt..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to install requirements"
    exit 1
fi

# Install additional MARL dependencies
echo "🧠 Installing additional MARL dependencies..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install wandb matplotlib seaborn moviepy gym==0.21.0 numpy==1.21.0 pyyaml pillow

# Install development dependencies
echo "🛠️  Installing development dependencies..."
pip install pytest pytest-cov black isort flake8

echo "✅ Python backend setup complete!"
echo ""

# =============================================================================
# PHASE 2: React Frontend Setup
# =============================================================================

echo "🌐 PHASE 2: Setting up React Frontend"
echo "====================================="

# Navigate to React frontend directory
cd react-frontend

# Check if Node.js is installed
if ! command -v npm &> /dev/null; then
    echo "❌ ERROR: Node.js/npm is not installed. Please install Node.js first."
    exit 1
fi

echo "📦 Installing React dependencies..."
npm install
if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to install React dependencies"
    exit 1
fi

echo "🔨 Building React app for production..."
CI=false npm run build
if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to build React app"
    exit 1
fi

echo "✅ React frontend build complete!"
echo ""

# Return to root directory
cd ..

# =============================================================================
# PHASE 3: Verification & Completion
# =============================================================================

echo "🔍 PHASE 3: Verification"
echo "========================"

# Check if key files exist
echo "📋 Verifying installation..."

if [ -d "venv" ]; then
    echo "✅ Python virtual environment: Created"
else
    echo "❌ Python virtual environment: Missing"
fi

if [ -d "react-frontend/build" ]; then
    echo "✅ React build directory: Created"
else
    echo "❌ React build directory: Missing"
fi

if [ -f "react-frontend/build/index.html" ]; then
    echo "✅ React index.html: Present"
else
    echo "❌ React index.html: Missing"
fi

echo ""
echo "🎉 EasyMARL Full Stack Deployment Complete!"
echo "==========================================="
echo ""
echo "🐍 Python Backend:"
echo "   • Virtual environment: ./venv/"
echo "   • To activate: source venv/bin/activate"
echo "   • To run backend: python flask_backend.py"
echo ""
echo "🌐 React Frontend:"
echo "   • Build directory: ./react-frontend/build/"
echo "   • To preview: cd react-frontend && npm start"
echo "   • To deploy: npm run deploy (from react-frontend/)"
echo ""
echo "📚 Next Steps:"
echo "   1. Activate Python environment: source venv/bin/activate"
echo "   2. Start Flask backend: python flask_backend.py"
echo "   3. In another terminal, start React: cd react-frontend && npm start"
echo "   4. Open browser to: http://localhost:3000"
echo ""
echo "🆘 Troubleshooting:"
echo "   • Backend issues: Check Python dependencies in requirements.txt"
echo "   • Frontend issues: Check Node.js version (requires 14+)"
echo "   • Port conflicts: Backend uses 5000, Frontend uses 3000"
echo ""
echo "✨ Happy training with EasyMARL!"
