"""
EasyMARL Flask Backend API
 
This Flask server provides REST API endpoints for the React frontend
to communicate with the Python training infrastructure. It maintains
the same functionality as the original GUI while enabling modern
web deployment on GitHub Pages.

Key Features:
- REST API endpoints for training control
- Real-time progress updates via polling
- Integration with existing Simple/Modern controllers
- WandB logging support
- Static file serving for React build
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import json
import threading
import time
import yaml
from datetime import datetime
import uuid

# Import existing EasyMARL components
import utils
from simple_multiagent_controller import SimpleMultiAgentController
from modern_multiagent_controller import ModernMultiAgentController
import torch

# Initialize Flask app
app = Flask(__name__, static_folder='react-frontend/build', static_url_path='')
CORS(app)  # Enable CORS for React development

# Global training state
training_sessions = {}
current_session = None

class TrainingSession:
    """
    Manages a single training session with progress tracking.
    
    This class wraps the existing controllers and provides
    thread-safe access to training progress for the API.
    """
    
    def __init__(self, session_id, config):
        self.session_id = session_id
        self.config = config
        self.controller = None
        self.training_thread = None
        self.is_training = False
        self.progress_data = {
            'episodes': [],
            'rewards': [],
            'lengths': [],
            'current_episode': 0,
            'status': 'ready'
        }
        self.start_time = None
        self.error = None
    
    def start_training(self):
        """Start training in background thread."""
        try:
            # Create environment
            env = utils.make_env(self.config['environment'])
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Create controller based on type
            if self.config['controllerType'] == 'simple':
                self.controller = SimpleMultiAgentController(
                    env=env,
                    config=self.config,
                    device=device,
                    algorithm=self.config['algorithm'],
                    training=True
                )
            else:
                self.controller = ModernMultiAgentController(
                    env=env,
                    config=self.config,
                    device=device,
                    algorithm=self.config['algorithm'],
                    training=True
                )
            
            # Start training thread
            self.training_thread = threading.Thread(target=self._training_loop)
            self.training_thread.daemon = True
            self.is_training = True
            self.start_time = datetime.now()
            self.progress_data['status'] = 'training'
            self.training_thread.start()
            
            return True
            
        except Exception as e:
            self.error = str(e)
            self.progress_data['status'] = 'error'
            return False
    
    def stop_training(self):
        """Stop training session."""
        self.is_training = False
        self.progress_data['status'] = 'stopped'
        
        if self.training_thread and self.training_thread.is_alive():
            # Signal to stop training
            self.is_training = False
    
    def _training_loop(self):
        """Main training loop running in background thread."""
        try:
            max_episodes = self.config.get('maxEpisodes', 1000)
            
            for episode in range(max_episodes):
                if not self.is_training:
                    break
                
                # Run one episode using existing controller
                episode_data = self.controller.run_one_episode(
                    episode=episode,
                    log=False,  # Disable console logging for API
                    train=True,
                    save_model=(episode % 100 == 0),
                    visualize=False
                )
                
                # Update progress data
                self.progress_data['episodes'].append(episode)
                self.progress_data['rewards'].append(episode_data['episode_reward'])
                self.progress_data['lengths'].append(episode_data['episode_length'])
                self.progress_data['current_episode'] = episode
                
                # Small delay to prevent overwhelming the system
                time.sleep(0.1)
            
            # Training completed
            self.is_training = False
            self.progress_data['status'] = 'completed'
            
            # Save final models
            if self.controller:
                self.controller.save_models(f"session_{self.session_id}_final")
                
        except Exception as e:
            self.error = str(e)
            self.is_training = False
            self.progress_data['status'] = 'error'


@app.route('/')
def serve_react_app():
    """Serve the React app for production deployment."""
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_react_static(path):
    """Serve static files for React app."""
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

# API Routes
@app.route('/api/training/start', methods=['POST'])
def start_training():
    """
    Start a new training session.
    
    Expected JSON payload:
    {
        "environment": "MultiGrid-Cluttered-Fixed-15x15",
        "algorithm": "qmix",
        "controllerType": "simple",
        "maxEpisodes": 1000,
        "learningRate": 0.001,
        "useWandB": false
    }
    """
    global current_session, training_sessions
    
    try:
        config = request.json
        
        # Stop existing session if running
        if current_session and current_session.is_training:
            current_session.stop_training()
        
        # Create new session
        session_id = str(uuid.uuid4())
        session = TrainingSession(session_id, config)
        
        # Start training
        if session.start_training():
            training_sessions[session_id] = session
            current_session = session
            
            return jsonify({
                'success': True,
                'sessionId': session_id,
                'message': f'Training started with {config["algorithm"].upper()} using {config["controllerType"]} controller'
            })
        else:
            return jsonify({
                'success': False,
                'error': session.error or 'Failed to start training'
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/training/stop', methods=['POST'])
def stop_training():
    """Stop the current training session."""
    global current_session
    
    try:
        if current_session and current_session.is_training:
            current_session.stop_training()
            return jsonify({
                'success': True,
                'message': 'Training stopped'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'No training session running'
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/training/progress', methods=['GET'])
def get_training_progress():
    """Get current training progress."""
    global current_session
    
    try:
        if current_session:
            return jsonify({
                'isTraining': current_session.is_training,
                'sessionId': current_session.session_id,
                'episode': current_session.progress_data['current_episode'],
                'episodes': current_session.progress_data['episodes'],
                'rewards': current_session.progress_data['rewards'],
                'lengths': current_session.progress_data['lengths'],
                'status': current_session.progress_data['status'],
                'error': current_session.error,
                'startTime': current_session.start_time.isoformat() if current_session.start_time else None
            })
        else:
            return jsonify({
                'isTraining': False,
                'sessionId': None,
                'episode': 0,
                'episodes': [],
                'rewards': [],
                'lengths': [],
                'status': 'ready',
                'error': None,
                'startTime': None
            })
            
    except Exception as e:
        return jsonify({
            'isTraining': False,
            'error': str(e)
        }), 500

@app.route('/api/algorithms', methods=['GET'])
def get_algorithms():
    """Get available algorithms with descriptions."""
    try:
        # Import algorithm descriptions from existing GUI
        from gui import ALGORITHM_DESCRIPTIONS
        
        algorithms = []
        for alg_name, description in ALGORITHM_DESCRIPTIONS.items():
            algorithms.append({
                'value': alg_name,
                'label': alg_name.upper(),
                'description': description.get('description', ''),
                'category': description.get('category', 'Unknown'),
                'use_case': description.get('use_case', ''),
                'pros': description.get('pros', []),
                'cons': description.get('cons', [])
            })
        
        return jsonify(algorithms)
        
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/api/environments', methods=['GET'])
def get_environments():
    """Get available environments with descriptions."""
    try:
        environments = [
            {
                'value': 'MultiGrid-Cluttered-Fixed-15x15',
                'label': 'MultiGrid Cluttered (15x15)',
                'description': 'Navigate through a cluttered 15x15 grid with obstacles',
                'agents': '2-4',
                'difficulty': 'Medium'
            },
            {
                'value': 'MultiGrid-Empty-8x8',
                'label': 'MultiGrid Empty (8x8)',
                'description': 'Simple navigation in an empty 8x8 grid',
                'agents': '2-4',
                'difficulty': 'Easy'
            },
            {
                'value': 'MultiGrid-FourRooms',
                'label': 'MultiGrid Four Rooms',
                'description': 'Navigate between connected rooms',
                'agents': '2-4',
                'difficulty': 'Hard'
            }
        ]
        
        return jsonify(environments)
        
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/api/training/download/<session_id>', methods=['GET'])
def download_training_data(session_id):
    """Download training data for a specific session."""
    try:
        if session_id in training_sessions:
            session = training_sessions[session_id]
            
            data = {
                'sessionId': session_id,
                'config': session.config,
                'progressData': session.progress_data,
                'startTime': session.start_time.isoformat() if session.start_time else None,
                'framework': 'EasyMARL React',
                'exportTime': datetime.now().isoformat()
            }
            
            return jsonify(data)
        else:
            return jsonify({
                'error': 'Session not found'
            }), 404
            
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'version': '1.0.0',
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    # Development server
    print("🚀 Starting EasyMARL Flask Backend")
    print("📱 React frontend should be built and available")
    print("🔗 API available at http://localhost:5000/api")
    print("🌐 Full app available at http://localhost:5000")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )
