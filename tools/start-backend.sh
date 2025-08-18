#!/bin/bash
# Start Flask Backend Only

echo "🐍 Starting EasyMARL Flask Backend..."

# Set Python path (go up one directory since we're in tools/)
export PYTHONPATH=$(dirname $PWD)
cd $(dirname $PWD)

# Start Flask backend
python api/flask_backend.py
