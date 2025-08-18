#!/usr/bin/env python3
"""
EasyMARL Flask Deployment Entry Point

This is the main entry point for deploying the EasyMARL Flask backend 
to cloud platforms like Vercel, Heroku, or Railway.
"""

import os
import sys

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the Flask app from the backend
from src.api.flask_backend import app

# Configure for production deployment
if __name__ == "__main__":
    # For local development
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
else:
    # For production deployment (Vercel, Heroku, etc.)
    # The app object is imported directly
    pass
