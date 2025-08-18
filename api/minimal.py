#!/usr/bin/env python3
"""
EasyMARL Minimal Flask API for Deployment

This is a lightweight version of the Flask API designed for cloud deployment
environments where installing heavy ML dependencies (PyTorch, etc.) may fail.

It provides basic API endpoints for frontend compatibility while gracefully
handling missing dependencies.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import json
import uuid
from datetime import datetime

# Initialize Flask app
app = Flask(__name__, static_folder='react-frontend/build', static_url_path='')
CORS(app)

# Serve React app
@app.route('/')
def serve_react_app():
    """Serve the React frontend"""
    try:
        return send_from_directory(app.static_folder, 'index.html')
    except Exception:
        return jsonify({"message": "EasyMARL API is running! Frontend not available."}), 200

@app.route('/<path:path>')
def serve_react_static(path):
    """Serve React static files"""
    try:
        return send_from_directory(app.static_folder, path)
    except Exception:
        return jsonify({"error": "File not found"}), 404

# Basic API endpoints for frontend compatibility
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "EasyMARL API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/algorithms', methods=['GET'])
def get_algorithms():
    """Get list of available algorithms"""
    algorithms = [
        {
            "name": "ippo",
            "display_name": "Independent PPO",
            "category": "Policy Gradient",
            "description": "Independent Proximal Policy Optimization",
            "supported": False,
            "reason": "Training dependencies not available in deployment environment"
        },
        {
            "name": "maddpg", 
            "display_name": "MADDPG",
            "category": "Actor-Critic",
            "description": "Multi-Agent Deep Deterministic Policy Gradient",
            "supported": False,
            "reason": "Training dependencies not available in deployment environment"
        },
        {
            "name": "qmix",
            "display_name": "QMIX", 
            "category": "Value-Based",
            "description": "Monotonic Value Function Factorisation",
            "supported": False,
            "reason": "Training dependencies not available in deployment environment"
        }
    ]
    
    return jsonify({
        "algorithms": algorithms,
        "note": "This is a minimal deployment version. For full training capabilities, run locally with complete dependencies.",
        "local_setup": "pip install -e . && python gui.py"
    })

@app.route('/api/environments', methods=['GET'])
def get_environments():
    """Get list of available environments"""
    environments = [
        {
            "name": "MultiGrid-Cluttered-Fixed-15x15",
            "display_name": "Cluttered Grid World",
            "description": "Multi-agent navigation in cluttered environment",
            "supported": False,
            "reason": "Environment dependencies not available in deployment environment"
        }
    ]
    
    return jsonify({
        "environments": environments,
        "note": "This is a minimal deployment version. For full environment support, run locally.",
        "local_setup": "pip install -e . && python gui.py"
    })

@app.route('/api/training/start', methods=['POST'])
def start_training():
    """Start training (minimal version returns info message)"""
    return jsonify({
        "success": False,
        "message": "Training not available in deployment environment",
        "note": "This is a minimal API version for demonstration. For actual training:",
        "instructions": [
            "1. Clone the repository: git clone https://github.com/shreyanmitra/EasyMARL",
            "2. Install locally: pip install -e .",
            "3. Run GUI: python gui.py",
            "4. Or run training directly: python main.py --algorithm ippo"
        ],
        "documentation": "https://github.com/shreyanmitra/EasyMARL#readme"
    }), 400

@app.route('/api/training/status', methods=['GET'])
def get_training_status():
    """Get training status"""
    return jsonify({
        "active": False,
        "message": "No training sessions active",
        "note": "Training not available in deployment environment"
    })

@app.route('/api/training/stop', methods=['POST'])
def stop_training():
    """Stop training"""
    return jsonify({
        "success": True,
        "message": "No training to stop"
    })

@app.route('/api/config/defaults', methods=['GET'])
def get_default_config():
    """Get default configuration"""
    return jsonify({
        "algorithm": "ippo",
        "environment": "MultiGrid-Cluttered-Fixed-15x15",
        "training": {
            "num_episodes": 1000,
            "learning_rate": 0.0003,
            "batch_size": 64
        },
        "note": "Default configuration for reference. Actual training requires local installation."
    })

@app.route('/api/info', methods=['GET'])
def get_api_info():
    """Get API information"""
    return jsonify({
        "name": "EasyMARL",
        "version": "1.0.0",
        "description": "Educational Multi-Agent Reinforcement Learning Framework",
        "author": "Shreyan Mitra",
        "repository": "https://github.com/shreyanmitra/EasyMARL",
        "deployment": {
            "type": "minimal",
            "capabilities": ["API documentation", "Algorithm information", "Frontend serving"],
            "limitations": ["No actual training", "No environment execution", "No model persistence"]
        },
        "local_installation": {
            "pip": "pip install easymarl",
            "github": "git clone https://github.com/shreyanmitra/EasyMARL && cd EasyMARL && pip install -e .",
            "gui": "python gui.py",
            "cli": "python main.py --help"
        }
    })

# Error handlers
@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        "error": "Endpoint not found",
        "available_endpoints": [
            "/api/health",
            "/api/algorithms", 
            "/api/environments",
            "/api/training/start",
            "/api/training/status",
            "/api/training/stop",
            "/api/config/defaults",
            "/api/info"
        ]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        "error": "Internal server error",
        "message": "Something went wrong on the server side"
    }), 500

if __name__ == "__main__":
    # For local development
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
