#!/bin/bash
# Auto-start script for Codespaces

echo "🎓 EasyMARL Auto-Starting in Codespaces..."

# Wait a moment for environment to be ready
sleep 3

# Start EasyMARL services in background
nohup bash tools/start-easymarl.sh > /tmp/easymarl.log 2>&1 &

# Wait for services to start
echo "⏳ Starting services (this takes ~15 seconds)..."
sleep 15

# Show status
echo "✅ EasyMARL services started!"

if [ -n "$CODESPACE_NAME" ]; then
    FRONTEND_URL="https://$CODESPACE_NAME-3000.app.github.dev"
    BACKEND_URL="https://$CODESPACE_NAME-5000.app.github.dev"
    
    echo ""
    echo "🌐 Your EasyMARL URLs:"
    echo "📱 Frontend: $FRONTEND_URL"
    echo "🔌 Backend:  $BACKEND_URL/api"
    echo ""
    echo "💡 The React app will auto-open in a new tab!"
    echo "📋 Check the 'Ports' tab in VS Code to see all services"
fi
