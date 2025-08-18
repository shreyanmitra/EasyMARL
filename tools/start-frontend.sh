#!/bin/bash
# Start React Frontend Only

echo "⚛️ Starting EasyMARL React Frontend..."

# Go to project root, then to React directory
cd $(dirname $PWD)/gui/react-frontend

# Set up environment for Codespaces if needed
if [ -n "$CODESPACE_NAME" ]; then
    echo "🎓 Configuring for Codespaces: $CODESPACE_NAME"
    BACKEND_URL="https://$CODESPACE_NAME-5000.app.github.dev"
    
    # Create local environment file
    cat > .env.local << EOF
REACT_APP_API_URL=$BACKEND_URL/api
REACT_APP_VERSION=1.0.0
REACT_APP_GITHUB_URL=https://github.com/shreyanmitra/EasyMARL
BROWSER=none
WDS_SOCKET_HOST=0.0.0.0
WDS_SOCKET_PORT=0
EOF
    
    echo "🔗 Backend URL set to: $BACKEND_URL/api"
fi

# Start React development server
npm start
