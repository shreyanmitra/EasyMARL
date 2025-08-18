#!/bin/bash
# EasyMARL Quick Start Script for Codespaces

echo "🎓 EasyMARL - GitHub Codespaces Quick Start"
echo "=========================================="
echo ""

# Get the project root directory (parent of tools/)
PROJECT_ROOT=$(dirname $PWD)
cd $PROJECT_ROOT

# Check if we're in Codespaces
if [ -n "$CODESPACE_NAME" ]; then
    echo "✅ Running in GitHub Codespaces: $CODESPACE_NAME"
    FRONTEND_URL="https://$CODESPACE_NAME-3000.app.github.dev"
    BACKEND_URL="https://$CODESPACE_NAME-5000.app.github.dev"
else
    echo "🏠 Running locally"
    FRONTEND_URL="http://localhost:3000"
    BACKEND_URL="http://localhost:5000"
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
check_service 5000 "Flask Backend"
check_service 3000 "React Frontend"

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
