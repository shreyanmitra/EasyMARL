#!/bin/bash
# EasyMARL Master Quick Start Script
# This is a simple wrapper that calls the actual script in tools/

echo "🚀 EasyMARL Quick Start"
echo "======================"

# Check if tools directory exists
if [ ! -d "tools" ]; then
    echo "❌ Error: tools directory not found"
    echo "💡 Make sure you're in the EasyMARL root directory"
    exit 1
fi

# Check if the main script exists
if [ ! -f "tools/start-easymarl.sh" ]; then
    echo "❌ Error: tools/start-easymarl.sh not found"
    echo "💡 Run the devcontainer setup first"
    exit 1
fi

# Make sure the script is executable
chmod +x tools/start-easymarl.sh tools/start-backend.sh tools/start-frontend.sh

# Run the main startup script
./tools/start-easymarl.sh
