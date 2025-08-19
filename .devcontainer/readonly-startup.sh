#!/bin/bash

echo "🔒 Starting EasyMARL Demo (Read-Only Mode)"
echo "======================================="

# Set read-only permissions on source files
echo "📁 Setting read-only permissions..."
find . -name "*.py" -not -path "./venv/*" -not -path "./.venv/*" -exec chmod 444 {} \;
find . -name "*.js" -not -path "./node_modules/*" -not -path "./gui/react-frontend/node_modules/*" -exec chmod 444 {} \;
find . -name "*.jsx" -not -path "./node_modules/*" -not -path "./gui/react-frontend/node_modules/*" -exec chmod 444 {} \;
find . -name "*.yaml" -not -path "./node_modules/*" -exec chmod 444 {} \;
find . -name "*.json" -not -path "./node_modules/*" -not -path "./gui/react-frontend/node_modules/*" -exec chmod 444 {} \;

# Create logs directory for runtime files
mkdir -p logs tmp

echo "🐍 Installing Python dependencies..."
pip install -r requirements.txt > logs/pip-install.log 2>&1

echo "🚀 Starting Flask backend..."
export FLASK_ENV=production
export DEMO_MODE=true
nohup python -m api.flask_backend > logs/flask.log 2>&1 &
FLASK_PID=$!

# Wait for Flask to start
echo "⏳ Waiting for Flask backend to start..."
for i in {1..30}; do
    if curl -s http://localhost:5000/api/health > /dev/null 2>&1; then
        echo "✅ Flask backend started successfully"
        break
    fi
    sleep 1
done

echo "⚛️  Installing React dependencies..."
cd gui/react-frontend
npm install > ../../logs/npm-install.log 2>&1

echo "🌐 Starting React frontend..."
export REACT_APP_DEMO_MODE=true
export REACT_APP_READ_ONLY=true
nohup npm start > ../../logs/react.log 2>&1 &
REACT_PID=$!
cd ../..

# Wait for React to start
echo "⏳ Waiting for React frontend to start..."
for i in {1..60}; do
    if curl -s http://localhost:3000 > /dev/null 2>&1; then
        echo "✅ React frontend started successfully"
        break
    fi
    sleep 1
done

echo ""
echo "🎉 EasyMARL Demo is ready!"
echo "======================================="
echo "🌐 React App: http://localhost:3000"
echo "🐍 Flask API: http://localhost:5000/api"
echo "📋 This is a READ-ONLY demonstration environment"
echo "📝 For development, use the main repository"
echo ""

# Save PIDs for cleanup
echo $FLASK_PID > logs/flask.pid
echo $REACT_PID > logs/react.pid

# Keep services running
wait
