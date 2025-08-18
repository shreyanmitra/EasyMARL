#!/bin/bash
# EasyMARL Codespaces Setup Script

echo "🚀 Setting up EasyMARL development environment..."

# Update system packages
sudo apt-get update && sudo apt-get install -y \
    build-essential \
    git \
    curl \
    wget \
    ffmpeg \
    htop \
    tree

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Install React dependencies  
echo "⚛️ Installing React dependencies..."
cd gui/react-frontend
npm install
cd ../..

# Set up environment variables for Codespaces
echo "🔧 Setting up environment variables..."
cp .devcontainer/.env.codespaces .env

# Create startup script
echo "📝 Creating startup script..."
cat > tools/start-easymarl.sh << 'EOF'
#!/bin/bash
echo "🎓 Starting EasyMARL in Codespaces..."

# Get Codespace name from environment
if [ -n "$CODESPACE_NAME" ]; then
    echo "📍 Codespace: $CODESPACE_NAME"
    FRONTEND_URL="https://$CODESPACE_NAME-3000.app.github.dev"
    BACKEND_URL="https://$CODESPACE_NAME-5000.app.github.dev"
    
    # Update React environment for Codespaces
    cd gui/react-frontend
    echo "REACT_APP_API_URL=$BACKEND_URL/api" > .env.local
    echo "REACT_APP_VERSION=1.0.0" >> .env.local
    echo "REACT_APP_GITHUB_URL=https://github.com/shreyanmitra/EasyMARL" >> .env.local
    cd ../..
    
    echo "🌐 URLs will be:"
    echo "   React Frontend: $FRONTEND_URL"
    echo "   Flask Backend:  $BACKEND_URL"
else
    echo "🏠 Running locally"
fi

# Start Flask backend in background
echo "🐍 Starting Flask backend..."
export PYTHONPATH=$PWD
python api/flask_backend.py &
FLASK_PID=$!

# Wait for Flask to start
echo "⏳ Waiting for Flask to start..."
sleep 8

# Start React frontend
echo "⚛️ Starting React frontend..."
cd gui/react-frontend
npm start &
REACT_PID=$!
cd ../..

echo ""
echo "✅ EasyMARL is running!"
if [ -n "$CODESPACE_NAME" ]; then
    echo "📱 Frontend: https://$CODESPACE_NAME-3000.app.github.dev"
    echo "🔌 Backend:  https://$CODESPACE_NAME-5000.app.github.dev/api"
else
    echo "📱 Frontend: http://localhost:3000"
    echo "🔌 Backend:  http://localhost:5000/api"
fi
echo "🛑 Press Ctrl+C to stop all services"
echo ""

# Function to cleanup processes
cleanup() {
    echo "🧹 Stopping services..."
    kill $FLASK_PID $REACT_PID 2>/dev/null
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup SIGINT SIGTERM

# Keep script running
wait
EOF

chmod +x tools/start-easymarl.sh

# Create individual service scripts
echo "📝 Creating individual service scripts..."

# Backend only script
cat > tools/start-backend.sh << 'EOF'
#!/bin/bash
echo "🐍 Starting Flask backend only..."
export PYTHONPATH=$PWD
python api/flask_backend.py
EOF

chmod +x tools/start-backend.sh

# Frontend only script
cat > tools/start-frontend.sh << 'EOF'
#!/bin/bash
echo "⚛️ Starting React frontend only..."
cd gui/react-frontend

# Set up Codespaces environment if needed
if [ -n "$CODESPACE_NAME" ]; then
    BACKEND_URL="https://$CODESPACE_NAME-5000.app.github.dev"
    echo "REACT_APP_API_URL=$BACKEND_URL/api" > .env.local
    echo "REACT_APP_VERSION=1.0.0" >> .env.local
    echo "REACT_APP_GITHUB_URL=https://github.com/shreyanmitra/EasyMARL" >> .env.local
fi

npm start
EOF

chmod +x tools/start-frontend.sh

echo ""
echo "✅ EasyMARL development environment setup complete!"
echo ""
echo "🎯 Quick Start Commands:"
echo "   ./tools/start-easymarl.sh     - Start both frontend and backend"
echo "   ./tools/start-backend.sh      - Start backend only"
echo "   ./tools/start-frontend.sh     - Start frontend only"
echo ""
echo "📚 Next Steps:"
echo "1. Run: ./tools/start-easymarl.sh"
echo "2. Wait for both services to start"
echo "3. Click the port notification links to access your app"
echo ""
