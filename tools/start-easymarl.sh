#!/bin/bash
# EasyMARL Quick Start Script for Codespaces

echo "🎓 EasyMARL - GitHub Codespaces Quick Start"
echo "=========================================="
echo ""

# Get the project root directory (parent of tools/)
PROJECT_ROOT=$(dirname $PWD)
cd $PROJECT_ROOT

# Check and install dependencies if needed
echo "🔍 Checking Python dependencies..."
if ! python -c "import flask" 2>/dev/null; then
    echo "📦 Flask not found - installing Python dependencies..."
    echo "⏳ This may take a few minutes for ML packages..."
    pip install --upgrade pip
    pip install -r requirements.txt
    echo "✅ Python dependencies installed!"
else
    echo "✅ Python dependencies are available"
fi

# Check React dependencies
if [ ! -d "gui/react-frontend/node_modules" ]; then
    echo "📦 Installing React dependencies..."
    cd gui/react-frontend
    npm install
    cd $PROJECT_ROOT
    echo "✅ React dependencies installed!"
else
    echo "✅ React dependencies are available"
fi

# Configure ports (environment variables override defaults)
FLASK_PORT=${PORT:-5000}
REACT_PORT=${REACT_PORT:-3000}

# Check if we're in Codespaces
if [ -n "$CODESPACE_NAME" ]; then
    echo "✅ Running in GitHub Codespaces: $CODESPACE_NAME"
    FRONTEND_URL="https://$CODESPACE_NAME-$REACT_PORT.app.github.dev"
    BACKEND_URL="https://$CODESPACE_NAME-$FLASK_PORT.app.github.dev"
else
    echo "🏠 Running locally"
    FRONTEND_URL="http://localhost:$REACT_PORT"
    BACKEND_URL="http://localhost:$FLASK_PORT"
fi

echo ""
echo "🚀 Starting EasyMARL services..."
echo ""

# Function to check if a service is running
check_service() {
    local port=$1
    local service_name=$2
    
    if curl -s "http://localhost:$port" > /dev/null 2>&1; then
        echo "✅ $service_name is running on port $port"
        return 0
    else
        echo "❌ $service_name is not running on port $port"
        return 1
    fi
}

# Start services in background
echo "🐍 Starting Flask backend..."
tools/start-backend.sh &
BACKEND_PID=$!

echo "⏳ Waiting for backend to start..."
sleep 10

echo "⚛️ Starting React frontend..."
tools/start-frontend.sh &
FRONTEND_PID=$!

echo "⏳ Waiting for frontend to start..."
sleep 15

echo ""
echo "🔍 Checking service status..."
check_service $FLASK_PORT "Flask Backend"
check_service $REACT_PORT "React Frontend"

echo ""
echo "🌐 Access URLs:"
echo "   📱 Frontend: $FRONTEND_URL"
echo "   🔌 Backend:  $BACKEND_URL/api"
echo ""

if [ -n "$CODESPACE_NAME" ]; then
    echo "💡 Click the port notifications in VS Code to open the URLs"
    echo "📋 Or use 'Ports' tab to see all forwarded ports"
else
    echo "💡 Open the URLs above in your browser"
fi

echo ""
echo "🛑 To stop all services: Ctrl+C or run 'pkill -f flask' and 'pkill -f react'"
echo "📚 For help: Check README.md or run individual scripts"
echo ""

# Keep script running and monitor services
trap 'echo "🧹 Stopping services..."; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0' SIGINT SIGTERM

wait
