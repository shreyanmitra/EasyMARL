"""
(C) Shreyan Mitra, based on starter code by Natasha Jaques

EasyMARL Flask Backend API
Modern Web Interface for Multi-Agent Reinforcement Learning

This Flask server provides REST API endpoints that allow the React frontend
to communicate with the Python MARL training infrastructure. It maintains
all the functionality of the original GUI for local development and GitHub Codespaces.

For MARL Beginners:
This is the bridge between the web interface (React) and the AI training code (Python).
You don't need to understand Flask to use EasyMARL, but this enables:
- Training agents through a web browser
- Real-time progress monitoring  
- Local development and GitHub Codespaces deployment

Key Features:
- REST API endpoints for training control (start, stop, configure)
- Real-time progress updates via polling
- Integration with existing Simple/Modern controllers
- Weights & Biases (WandB) logging support
- Static file serving for React build
- Thread-safe training session management

API Endpoints:
- GET /api/algorithms - List available MARL algorithms
- POST /api/training/start - Start training with specified configuration  
- GET /api/training/status - Get current training progress
- POST /api/training/stop - Stop current training session
- GET /api/config/defaults - Get default configuration for algorithms

Architecture:
React Frontend ↔ Flask API ↔ MARL Controllers ↔ Training Algorithms

For Developers:
The Flask server runs the actual MARL training in background threads while
serving the React frontend and providing API access to training status.
"""

# Import Flask components for web server functionality
from flask import Flask, request, jsonify, send_from_directory  # Core Flask functionality
from flask_cors import CORS                                    # Cross-Origin Resource Sharing
from flask_limiter import Limiter                             # Rate limiting
from flask_limiter.util import get_remote_address           # Client IP detection
import os          # Operating system interface
import json        # JSON data handling
import threading   # Multi-threading support
import time        # Time-related functions
import yaml        # YAML configuration file parsing
from datetime import datetime  # Date and time handling
import uuid        # UUID generation for session IDs

# Import EasyMARL components - Updated for new directory structure
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Core utilities (always available)
try:
    from easymarl.core.utils import *
except ImportError:
    utils = None

# Training components (imported only when needed)
_modern_controller = None
_simple_controller = None
_research_interface = None
_config_manager = None
_torch = None

def get_modern_controller():
    """Lazy import of ModernMultiAgentController"""
    global _modern_controller
    if _modern_controller is None:
        try:
            from easymarl.controllers.modern_multiagent_controller import ModernMultiAgentController
            _modern_controller = ModernMultiAgentController
        except ImportError:
            _modern_controller = None
    return _modern_controller

def get_simple_controller():
    """Lazy import of SimpleMultiAgentController"""  
    global _simple_controller
    if _simple_controller is None:
        try:
            from easymarl.controllers.simple_multiagent_controller import SimpleMultiAgentController
            _simple_controller = SimpleMultiAgentController
        except ImportError:
            _simple_controller = None
    return _simple_controller

def get_research_interface():
    """Lazy import of research interface"""
    global _research_interface
    if _research_interface is None:
        try:
            from easymarl.core.research_interface import get_research_interface as _get_research
            _research_interface = _get_research
        except ImportError:
            _research_interface = None
    return _research_interface

def get_config_manager():
    """Lazy import of config manager"""
    global _config_manager
    if _config_manager is None:
        try:
            from easymarl.core.config_manager import get_config_manager, ExperimentConfig, AgentConfig, EnvironmentConfig, TrainingConfig
            _config_manager = (get_config_manager, ExperimentConfig, AgentConfig, EnvironmentConfig, TrainingConfig)
        except ImportError:
            _config_manager = None
    return _config_manager

def get_torch():
    """Lazy import of PyTorch"""
    global _torch
    if _torch is None:
        try:
            import torch
            _torch = torch
        except ImportError:
            _torch = None
    return _torch

# Initialize Flask web application
app = Flask(__name__, static_folder='react-frontend/build', static_url_path='')

# Demo Mode Detection
DEMO_MODE = os.environ.get('DEMO_MODE', 'false').lower() == 'true'
READ_ONLY = os.environ.get('READ_ONLY', 'false').lower() == 'true'

# Security Configuration
import secrets
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', secrets.token_hex(16))
app.config['JSON_SORT_KEYS'] = False  # Don't expose internal structure
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False  # Minimize response size

# Configure CORS dynamically based on environment
CODESPACE_NAME = os.environ.get('CODESPACE_NAME')
if CODESPACE_NAME:
    # Codespaces-specific CORS configuration
    allowed_origins = [
        f"https://{CODESPACE_NAME}-3000.app.github.dev",
        "http://localhost:3000",
        "https://shreyanmitra.github.io"
    ]
    CORS(app, origins=allowed_origins)
else:
    # Default CORS configuration for local development
    CORS(app)

# Rate Limiting Configuration
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["1000 per hour", "100 per minute"],
    storage_uri="memory://",  # Use in-memory storage for simplicity
    strategy="fixed-window"
)

# Security Functions
def validate_algorithm(algorithm):
    """Validate algorithm parameter against whitelist"""
    allowed_algorithms = [
        'qmix', 'vdn', 'qtran', 'mappo', 'maddpg', 'coma', 'ippo', 'iql', 
        'maven', 'hql', 'lql', 'mfq', 'nfsp', 'dcg', 'maacc', 'wolfphc',
        'comacomm', 'maddpgcomm', 'minimaxq', 'ppo'
    ]
    return algorithm.lower() in allowed_algorithms

def validate_environment(environment):
    """Validate environment parameter against whitelist"""
    allowed_environments = [
        'MultiGrid-Empty-6x6-v0', 'MultiGrid-Empty-8x8-v0', 
        'MultiGrid-Empty-16x16-v0', 'MultiGrid-FourRooms-v0',
        'MultiGrid-Cluttered-Fixed-15x15', 'MultiGrid-Cluttered-v0',
        'MultiGrid-DoorKey-5x5-v0', 'MultiGrid-DoorKey-6x6-v0',
        'MultiGrid-DoorKey-8x8-v0'
    ]
    return environment in allowed_environments

def validate_input_data(data, required_fields=None, max_length=1000):
    """Validate and sanitize input data"""
    if not isinstance(data, dict):
        return False, "Invalid JSON data"
    
    if required_fields:
        for field in required_fields:
            if field not in data:
                return False, f"Missing required field: {field}"
            
            # Basic string length validation
            if isinstance(data[field], str) and len(data[field]) > max_length:
                return False, f"Field {field} too long (max {max_length} chars)"
    
    return True, "Valid"

# Enhanced Error Handlers
@app.errorhandler(400)
def bad_request(error):
    return jsonify({'success': False, 'error': 'Bad request', 'message': str(error)}), 400

@app.errorhandler(500)
def internal_error(error):
    # Don't expose internal errors in production
    is_production = os.environ.get('FLASK_ENV') == 'production'
    if is_production:
        return jsonify({'success': False, 'error': 'Internal server error'}), 500
    else:
        return jsonify({'success': False, 'error': 'Internal server error', 'details': str(error)}), 500

# Demo Mode Decorator
def demo_mode_check(f):
    """Decorator to block training operations in demo mode"""
    from functools import wraps
    
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if DEMO_MODE or READ_ONLY:
            return jsonify({
                'success': False, 
                'error': 'Demo Mode',
                'message': 'This is a read-only demonstration. Training is disabled.',
                'demo_mode': True
            }), 403
        return f(*args, **kwargs)
    return decorated_function

# Global training state management
# These variables maintain the state across API requests
training_sessions = {}  # Dictionary to store multiple training sessions
current_session = None  # Reference to currently active training session

class TrainingSession:
    """
    Manages a single MARL training session with progress tracking.
    
    This class wraps the existing controllers (Simple/Modern) and provides
    thread-safe access to training progress for the web API. Each training
    session runs in its own thread to prevent blocking the web server.
    
    Key Responsibilities:
    1. Execute MARL training in background thread
    2. Track training progress (episodes, rewards, losses)
    3. Provide thread-safe status updates for API
    4. Manage training lifecycle (start, stop, cleanup)
    
    For MARL Beginners:
    Think of this as a "training manager" that handles one training experiment.
    It runs the AI training in the background while letting you check progress
    through the web interface.
    """
    
    def __init__(self, session_id, config):
        """
        Initialize a new training session with the specified configuration.
        
        Args:
            session_id (str): Unique identifier for this training session
            config (dict): Training configuration including algorithm, environment, etc.
                          Example: {'algorithm': 'ippo', 'env_name': 'MultiGrid-Cluttered-Fixed-15x15'}
        
        For Beginners:
        This sets up everything needed to run a MARL training experiment
        based on the configuration from the web interface.
        """
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
            if utils is None:
                raise ImportError("Utils module not available")
                
            env = utils.make_env(self.config['environment'])
            
            # Get device (with fallback if torch not available)
            torch_module = get_torch()
            if torch_module:
                device = torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")
            else:
                device = "cpu"  # Fallback to CPU if torch not available
            
            # Create controller based on type
            if self.config['controllerType'] == 'simple':
                SimpleController = get_simple_controller()
                if SimpleController is None:
                    raise ImportError("SimpleMultiAgentController not available")
                self.controller = SimpleController(
                    env=env,
                    config=self.config,
                    device=device,
                    algorithm=self.config['algorithm'],
                    training=True
                )
            else:
                ModernController = get_modern_controller()
                if ModernController is None:
                    raise ImportError("ModernMultiAgentController not available")
                self.controller = ModernController(
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
@limiter.limit("5 per minute", error_message="Training start requests are limited to 5 per minute")
@demo_mode_check
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
        
        # Input validation
        if not config:
            return jsonify({'success': False, 'error': 'No configuration provided'}), 400
        
        # Validate required fields
        required_fields = ['environment', 'algorithm', 'controllerType']
        for field in required_fields:
            if field not in config:
                return jsonify({'success': False, 'error': f'Missing required field: {field}'}), 400
        
        # Validate specific fields
        if not validate_environment(config['environment']):
            return jsonify({'success': False, 'error': 'Invalid environment'}), 400
        
        if not validate_algorithm(config['algorithm']):
            return jsonify({'success': False, 'error': 'Invalid algorithm'}), 400
        
        # Validate controller type
        valid_controllers = ['simple', 'modern', 'vectorized']
        if config['controllerType'] not in valid_controllers:
            return jsonify({'success': False, 'error': f'Invalid controller type. Must be one of: {valid_controllers}'}), 400
        
        # Validate numeric parameters
        if 'maxEpisodes' in config:
            try:
                max_episodes = int(config['maxEpisodes'])
                if max_episodes <= 0 or max_episodes > 50000:
                    return jsonify({'success': False, 'error': 'maxEpisodes must be between 1 and 50000'}), 400
                config['maxEpisodes'] = max_episodes
            except (ValueError, TypeError):
                return jsonify({'success': False, 'error': 'maxEpisodes must be a valid integer'}), 400
        
        if 'learningRate' in config:
            try:
                lr = float(config['learningRate'])
                if lr <= 0 or lr > 1:
                    return jsonify({'success': False, 'error': 'learningRate must be between 0 and 1'}), 400
                config['learningRate'] = lr
            except (ValueError, TypeError):
                return jsonify({'success': False, 'error': 'learningRate must be a valid number'}), 400
        
        # Validate boolean parameters
        if 'useWandB' in config:
            if not isinstance(config['useWandB'], bool):
                return jsonify({'success': False, 'error': 'useWandB must be true or false'}), 400
        
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
@limiter.limit("10 per minute")
@demo_mode_check
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
        'demo_mode': DEMO_MODE,
        'read_only': READ_ONLY,
        'training_disabled': DEMO_MODE or READ_ONLY,
        'timestamp': datetime.now().isoformat()
    })

# =============================================================================
# MANUAL CONTROL API ENDPOINTS
# Manual environment control functionality from manual_control_multigrid.py
# =============================================================================

# Global state for manual control session
manual_control_session = {
    'env': None,
    'state': None,
    'step_count': 0,
    'total_reward': 0,
    'is_initialized': False,
    'agents': []
}

@app.route('/api/manual-control/init', methods=['POST'])
def init_manual_control():
    """Initialize manual control environment."""
    global manual_control_session
    
    try:
        data = request.get_json()
        env_name = data.get('environment', 'MultiGrid-Cluttered-Fixed-15x15')
        
        # Import necessary modules
        import gymnasium as gym
        from envs import gym_multigrid
        from easymarl.environments.gym_multigrid import multigrid_envs
        
        # Create environment
        env = gym.make(env_name)
        state = env.reset()
        
        # Extract agent information
        agents = []
        for i in range(env.n_agents):
            agent_info = {
                'id': i,
                'position': None,  # Will be updated from state
                'direction': None,
                'carrying': None
            }
            
            # Try to extract position if available
            if hasattr(env, 'agents') and i < len(env.agents):
                agent = env.agents[i]
                if hasattr(agent, 'pos'):
                    agent_info['position'] = {'x': agent.pos[0], 'y': agent.pos[1]}
                if hasattr(agent, 'dir'):
                    agent_info['direction'] = agent.dir
                if hasattr(agent, 'carrying'):
                    agent_info['carrying'] = str(agent.carrying) if agent.carrying else None
            
            agents.append(agent_info)
        
        # Create environment state for visualization
        env_state = {
            'width': getattr(env, 'width', 15),
            'height': getattr(env, 'height', 15),
            'grid': []
        }
        
        # Try to extract grid information
        if hasattr(env, 'grid'):
            for y in range(env_state['height']):
                for x in range(env_state['width']):
                    try:
                        cell = env.grid.get(x, y)
                        cell_info = {
                            'type': cell.__class__.__name__ if cell else 'Empty',
                            'color': getattr(cell, 'color', '#ffffff') if cell else '#ffffff',
                            'emoji': '🧱' if cell and cell.__class__.__name__ == 'Wall' else '',
                            'char': str(cell) if cell else ' '
                        }
                        env_state['grid'].append(cell_info)
                    except:
                        env_state['grid'].append({
                            'type': 'Unknown',
                            'color': '#ffffff',
                            'emoji': '',
                            'char': ' '
                        })
        
        # Update global session
        manual_control_session.update({
            'env': env,
            'state': state,
            'step_count': 0,
            'total_reward': 0,
            'is_initialized': True,
            'agents': agents
        })
        
        return jsonify({
            'success': True,
            'environment': env_name,
            'state': env_state,
            'agents': agents,
            'n_agents': env.n_agents
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to initialize environment: {str(e)}'
        }), 500

@app.route('/api/manual-control/step', methods=['POST'])
def manual_control_step():
    """Execute a single step in manual control mode."""
    global manual_control_session
    
    try:
        if not manual_control_session['is_initialized']:
            return jsonify({
                'success': False,
                'error': 'Environment not initialized'
            }), 400
        
        data = request.get_json()
        agent_id = data.get('agent_id', 0)
        action = data.get('action', 0)
        
        env = manual_control_session['env']
        
        # Create action array for all agents (others stay/noop)
        actions = [0] * env.n_agents  # 0 = stay action
        actions[agent_id] = action
        
        # Execute step
        next_state, rewards, done, info = env.step(actions)
        
        # Update session state
        manual_control_session['state'] = next_state
        manual_control_session['step_count'] += 1
        
        # Calculate total reward
        step_reward = sum(rewards) if isinstance(rewards, (list, tuple)) else rewards
        manual_control_session['total_reward'] += step_reward
        
        # Update agent positions
        for i, agent_info in enumerate(manual_control_session['agents']):
            if hasattr(env, 'agents') and i < len(env.agents):
                agent = env.agents[i]
                if hasattr(agent, 'pos'):
                    agent_info['position'] = {'x': agent.pos[0], 'y': agent.pos[1]}
                if hasattr(agent, 'dir'):
                    agent_info['direction'] = agent.dir
                if hasattr(agent, 'carrying'):
                    agent_info['carrying'] = str(agent.carrying) if agent.carrying else None
        
        # Create updated environment state
        env_state = {
            'width': getattr(env, 'width', 15),
            'height': getattr(env, 'height', 15),
            'grid': []
        }
        
        # Update grid visualization
        if hasattr(env, 'grid'):
            for y in range(env_state['height']):
                for x in range(env_state['width']):
                    try:
                        cell = env.grid.get(x, y)
                        cell_info = {
                            'type': cell.__class__.__name__ if cell else 'Empty',
                            'color': getattr(cell, 'color', '#ffffff') if cell else '#ffffff',
                            'emoji': '🧱' if cell and cell.__class__.__name__ == 'Wall' else '',
                            'char': str(cell) if cell else ' '
                        }
                        env_state['grid'].append(cell_info)
                    except:
                        env_state['grid'].append({
                            'type': 'Unknown',
                            'color': '#ffffff',
                            'emoji': '',
                            'char': ' '
                        })
        
        return jsonify({
            'success': True,
            'state': env_state,
            'step_count': manual_control_session['step_count'],
            'total_reward': manual_control_session['total_reward'],
            'step_reward': step_reward,
            'actions': actions,
            'done': done,
            'agents': manual_control_session['agents']
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Step execution failed: {str(e)}'
        }), 500

@app.route('/api/manual-control/reset', methods=['POST'])
def manual_control_reset():
    """Reset the manual control environment."""
    global manual_control_session
    
    try:
        if not manual_control_session['is_initialized']:
            return jsonify({
                'success': False,
                'error': 'Environment not initialized'
            }), 400
        
        env = manual_control_session['env']
        state = env.reset()
        
        # Reset session state
        manual_control_session.update({
            'state': state,
            'step_count': 0,
            'total_reward': 0
        })
        
        # Update agent positions
        for i, agent_info in enumerate(manual_control_session['agents']):
            if hasattr(env, 'agents') and i < len(env.agents):
                agent = env.agents[i]
                if hasattr(agent, 'pos'):
                    agent_info['position'] = {'x': agent.pos[0], 'y': agent.pos[1]}
                if hasattr(agent, 'dir'):
                    agent_info['direction'] = agent.dir
                if hasattr(agent, 'carrying'):
                    agent_info['carrying'] = None
        
        # Create environment state
        env_state = {
            'width': getattr(env, 'width', 15),
            'height': getattr(env, 'height', 15),
            'grid': []
        }
        
        # Update grid
        if hasattr(env, 'grid'):
            for y in range(env_state['height']):
                for x in range(env_state['width']):
                    try:
                        cell = env.grid.get(x, y)
                        cell_info = {
                            'type': cell.__class__.__name__ if cell else 'Empty',
                            'color': getattr(cell, 'color', '#ffffff') if cell else '#ffffff',
                            'emoji': '🧱' if cell and cell.__class__.__name__ == 'Wall' else '',
                            'char': str(cell) if cell else ' '
                        }
                        env_state['grid'].append(cell_info)
                    except:
                        env_state['grid'].append({
                            'type': 'Unknown',
                            'color': '#ffffff',
                            'emoji': '',
                            'char': ' '
                        })
        
        return jsonify({
            'success': True,
            'state': env_state,
            'agents': manual_control_session['agents']
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Reset failed: {str(e)}'
        }), 500

@app.route('/api/manual-control/status', methods=['GET'])
def manual_control_status():
    """Get current manual control session status."""
    global manual_control_session
    
    return jsonify({
        'is_initialized': manual_control_session['is_initialized'],
        'step_count': manual_control_session['step_count'],
        'total_reward': manual_control_session['total_reward'],
        'n_agents': len(manual_control_session['agents'])
    })

if __name__ == '__main__':
    # Development server
    print("🚀 Starting EasyMARL Flask Backend")
# =============================================================================
# RESEARCH API ENDPOINTS
# Enhanced endpoints for research-oriented features
# =============================================================================

@app.route('/api/research/algorithms/browse', methods=['GET'])
def get_algorithm_browser():
    """Get structured algorithm browser data for research interface."""
    try:
        research = get_research_interface()
        browser_data = research.get_algorithm_browser()
        
        return jsonify({
            'success': True,
            'data': browser_data
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/research/algorithms/<algorithm>/info', methods=['GET'])
def get_algorithm_details(algorithm):
    """Get detailed information about a specific algorithm."""
    try:
        research = get_research_interface()
        algo_info = research.discovery.get_algorithm_info(algorithm)
        
        if not algo_info:
            return jsonify({
                'success': False,
                'error': f'Algorithm {algorithm} not found'
            }), 404
        
        return jsonify({
            'success': True,
            'algorithm': algorithm,
            'data': algo_info
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/research/algorithms/<algorithm>/parameters', methods=['GET'])
def get_algorithm_parameters(algorithm):
    """Get parameter schema for a specific algorithm."""
    try:
        research = get_research_interface()
        schema = research.get_parameter_schema(algorithm)
        
        return jsonify({
            'success': True,
            'algorithm': algorithm,
            'schema': schema
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/research/algorithms/compare', methods=['POST'])
def compare_algorithms():
    """Compare multiple algorithms across key dimensions."""
    try:
        data = request.get_json()
        algorithms = data.get('algorithms', [])
        
        research = get_research_interface()
        comparison = research.discovery.compare_algorithms(algorithms)
        
        return jsonify({
            'success': True,
            'comparison': comparison
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/research/recommendations', methods=['POST'])
def get_algorithm_recommendations():
    """Get algorithm recommendations based on research focus."""
    try:
        data = request.get_json()
        research_focus = data.get('research_focus', 'cooperative_ai')
        experience_level = data.get('experience_level', 'intermediate')
        
        research = get_research_interface()
        recommendations = research.discovery.get_research_recommendations(
            research_focus, experience_level
        )
        
        return jsonify({
            'success': True,
            'recommendations': recommendations
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/research/experiments/create', methods=['POST'])
def create_research_experiment():
    """Create a new research experiment configuration."""
    try:
        data = request.get_json()
        
        # Extract experiment parameters
        name = data.get('name', f'experiment_{int(time.time())}')
        description = data.get('description', '')
        algorithm = data.get('algorithm', 'ippo')
        environment = data.get('environment', 'MultiGrid-Cluttered-Fixed-15x15')
        
        research = get_research_interface()
        config = research.create_experiment(name, description, algorithm, environment)
        
        # Update configuration with provided parameters
        if 'total_episodes' in data:
            config.total_episodes = data['total_episodes']
        if 'n_agents' in data:
            config.n_agents = data['n_agents']
        if 'agent_configs' in data:
            # Update agent configurations
            for i, agent_data in enumerate(data['agent_configs']):
                if i < len(config.agent_configs):
                    for key, value in agent_data.items():
                        if hasattr(config.agent_configs[i], key):
                            setattr(config.agent_configs[i], key, value)
        
        # Validate configuration
        warnings = research.validate_configuration(config)
        
        return jsonify({
            'success': True,
            'config': config.to_dict(),
            'warnings': warnings
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/research/experiments/save', methods=['POST'])
def save_experiment_config():
    """Save experiment configuration to file."""
    try:
        data = request.get_json()
        config_data = data.get('config')
        filename = data.get('filename', f'experiment_{int(time.time())}.json')
        
        # Create config object
        config = ExperimentConfig.from_dict(config_data)
        
        # Save to file
        filepath = os.path.join('experiments', filename)
        research = get_research_interface()
        research.save_experiment_config(config, filepath)
        
        return jsonify({
            'success': True,
            'filepath': filepath
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/research/experiments/load', methods=['POST'])
def load_experiment_config():
    """Load experiment configuration from file."""
    try:
        data = request.get_json()
        filepath = data.get('filepath')
        
        research = get_research_interface()
        config = research.load_experiment_config(filepath)
        
        return jsonify({
            'success': True,
            'config': config.to_dict()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/research/hyperparameter-search/start', methods=['POST'])
def start_hyperparameter_search():
    """Start hyperparameter optimization."""
    try:
        data = request.get_json()
        config_data = data.get('config')
        
        # Create config object
        config = ExperimentConfig.from_dict(config_data)
        config.hyperparameter_search = True
        config.search_budget = data.get('search_budget', 50)
        
        research = get_research_interface()
        
        # Start optimization in background thread
        def run_optimization():
            try:
                result = research.start_hyperparameter_search(config)
                # Store result somewhere accessible
                global optimization_results
                if 'optimization_results' not in globals():
                    optimization_results = {}
                optimization_results[config.name] = result
            except Exception as e:
                print(f"Optimization failed: {e}")
        
        optimization_thread = threading.Thread(target=run_optimization)
        optimization_thread.daemon = True
        optimization_thread.start()
        
        return jsonify({
            'success': True,
            'message': 'Hyperparameter optimization started',
            'experiment_name': config.name
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/research/hyperparameter-search/status/<experiment_name>', methods=['GET'])
def get_optimization_status(experiment_name):
    """Get status of hyperparameter optimization."""
    try:
        global optimization_results
        if 'optimization_results' not in globals():
            optimization_results = {}
        
        if experiment_name in optimization_results:
            return jsonify({
                'success': True,
                'status': 'completed',
                'result': optimization_results[experiment_name]
            })
        else:
            return jsonify({
                'success': True,
                'status': 'running',
                'message': 'Optimization in progress'
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/research/taxonomy', methods=['GET'])
def get_algorithm_taxonomy():
    """Get the complete algorithm taxonomy for visualization."""
    try:
        research = get_research_interface()
        taxonomy_tree = research.discovery.taxonomy.get_taxonomy_tree()
        learning_progression = research.discovery.taxonomy.get_learning_progression()
        
        return jsonify({
            'success': True,
            'taxonomy': taxonomy_tree,
            'learning_progression': learning_progression
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# =============================================================================
# ENHANCED TRAINING API WITH RESEARCH FEATURES
# =============================================================================

@app.route('/api/research/training/start', methods=['POST'])
def start_research_training():
    """Start training with research-oriented configuration."""
    try:
        data = request.get_json()
        config_data = data.get('config')
        
        # Create experiment config
        config = ExperimentConfig.from_dict(config_data)
        
        # Convert to utils config format
        utils_config = {
            'algorithm': config.algorithm,
            'environment': config.environment,
            'episodes': config.total_episodes,
            'max_steps': config.max_steps_per_episode,
            'device': config.device,
            'seed': config.seed,
            'use_wandb': config.use_wandb,
            'wandb_project': config.wandb_project,
            'experiment_name': config.name
        }
        
        # Add agent-specific parameters
        if config.agent_configs:
            agent_config = config.agent_configs[0]  # Use first agent as template
            utils_config.update({
                'lr': agent_config.learning_rate,
                'gamma': agent_config.gamma,
                'batch_size': agent_config.batch_size,
                'hidden_dims': agent_config.hidden_dims
            })
            
            # Algorithm-specific parameters
            if config.algorithm == 'ippo':
                utils_config.update({
                    'clip_ratio': agent_config.clip_ratio,
                    'entropy_coef': agent_config.entropy_coef,
                    'gae_lambda': agent_config.gae_lambda
                })
        
        # Start training using existing training infrastructure
        session_id = str(uuid.uuid4())
        
        def training_worker():
            try:
                # Check for required dependencies
                if utils is None:
                    raise ImportError("Utils module not available")
                    
                # Create environment
                env = utils.make_env(utils_config)
                
                # Get device (with fallback if torch not available)
                torch_module = get_torch()
                if torch_module:
                    device = torch_module.device('cuda' if torch_module.cuda.is_available() and utils_config.get('device') != 'cpu' else 'cpu')
                else:
                    device = 'cpu'  # Fallback to CPU if torch not available
                
                # Create controller
                ModernController = get_modern_controller()
                if ModernController is None:
                    raise ImportError("ModernMultiAgentController not available")
                    
                controller = ModernController(
                    env=env,
                    config=utils_config,
                    device=device,
                    algorithm=config.algorithm,
                    training=True,
                    debug=False
                )
                
                # Store session
                training_sessions[session_id] = {
                    'controller': controller,
                    'config': utils_config,
                    'experiment_config': config,
                    'status': 'running',
                    'start_time': datetime.now(),
                    'progress': {
                        'episode': 0,
                        'total_episodes': config.total_episodes,
                        'rewards': [],
                        'episode_lengths': []
                    }
                }
                
                # Start training
                controller.train(config.total_episodes)
                
                # Update status
                training_sessions[session_id]['status'] = 'completed'
                training_sessions[session_id]['end_time'] = datetime.now()
                
            except Exception as e:
                if session_id in training_sessions:
                    training_sessions[session_id]['status'] = 'failed'
                    training_sessions[session_id]['error'] = str(e)
                print(f"Training failed: {e}")
        
        # Start training thread
        training_thread = threading.Thread(target=training_worker)
        training_thread.daemon = True
        training_thread.start()
        
        global current_session
        current_session = session_id
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'message': 'Research training started',
            'config': config.to_dict()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# Configuration Management Routes
@app.route('/api/config/create', methods=['POST'])
def create_config():
    """Create a new experiment configuration."""
    try:
        data = request.json
        algorithm = data.get('algorithm')
        n_agents = data.get('n_agents', 2)
        
        config_manager = get_config_manager()
        config = config_manager.create_default_config(algorithm, n_agents)
        
        # Apply any custom parameters
        if 'custom_params' in data:
            # Apply custom parameters to the config
            for key, value in data['custom_params'].items():
                config_manager._set_nested_param(config, key, value)
        
        # Save the configuration
        config_path = config_manager.save_config(config)
        
        return jsonify({
            'success': True,
            'config_path': config_path,
            'config': {
                'experiment_name': config.experiment_name,
                'algorithm': config.algorithm,
                'n_agents': len(config.agents),
                'description': config.description
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/config/load/<config_name>', methods=['GET'])
def load_config_api(config_name):
    """Load an experiment configuration."""
    try:
        config_manager = get_config_manager()
        config = config_manager.load_config(f"config/experiments/{config_name}.yaml")
        
        return jsonify({
            'success': True,
            'config': {
                'experiment_name': config.experiment_name,
                'algorithm': config.algorithm,
                'description': config.description,
                'n_agents': len(config.agents),
                'environment': {
                    'env_name': config.environment.env_name,
                    'size': config.environment.size,
                    'max_steps': config.environment.max_steps
                },
                'training': {
                    'max_episodes': config.training.max_episodes,
                    'eval_interval': config.training.eval_interval,
                    'use_wandb': config.training.use_wandb
                },
                'agents': [
                    {
                        'agent_id': agent.agent_id,
                        'learning_rate': agent.learning_rate,
                        'gamma': agent.gamma,
                        'epsilon': agent.epsilon,
                        'hidden_dims': agent.hidden_dims
                    } for agent in config.agents
                ]
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/config/list', methods=['GET'])
def list_configs_api():
    """List available configurations."""
    try:
        config_manager = get_config_manager()
        
        configs = {
            'experiments': config_manager.list_experiments(),
            'templates': config_manager.list_templates(),
            'presets': {}
        }
        
        # Get presets for each algorithm
        research_interface = get_research_interface()
        algorithms = research_interface.discovery.get_available_algorithms()
        
        for algo in algorithms:
            configs['presets'][algo] = config_manager.list_presets(algo)
        
        return jsonify({'success': True, 'configs': configs})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/config/validate', methods=['POST'])
def validate_config_api():
    """Validate an experiment configuration."""
    try:
        config_data = request.json
        
        # Convert to ExperimentConfig
        agents = [AgentConfig(**agent) for agent in config_data['agents']]
        environment = EnvironmentConfig(**config_data['environment'])
        training = TrainingConfig(**config_data['training'])
        
        config = ExperimentConfig(
            experiment_name=config_data['experiment_name'],
            algorithm=config_data['algorithm'],
            environment=environment,
            training=training,
            agents=agents,
            description=config_data.get('description', '')
        )
        
        config_manager = get_config_manager()
        issues = config_manager.validate_config(config)
        
        return jsonify({
            'success': True,
            'valid': len(issues) == 0,
            'issues': issues
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/config/algorithm-parameters/<algorithm>', methods=['GET'])
def get_algorithm_parameters(algorithm):
    """Get all configurable parameters for a specific algorithm."""
    try:
        config_manager = get_config_manager()
        parameters = config_manager.get_algorithm_parameters(algorithm)
        
        return jsonify({
            'success': True,
            'algorithm': algorithm,
            'parameters': parameters
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/config/algorithm-template/<algorithm>', methods=['GET'])
def get_algorithm_template(algorithm):
    """Get the parameter template for a specific algorithm."""
    try:
        from config_manager import AlgorithmTemplates
        template = AlgorithmTemplates.get_algorithm_template(algorithm)
        
        return jsonify({
            'success': True,
            'algorithm': algorithm,
            'template': template
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/config/create-from-template', methods=['POST'])
def create_config_from_template():
    """Create configuration from algorithm template with custom parameters."""
    try:
        data = request.json
        algorithm = data.get('algorithm')
        n_agents = data.get('n_agents', 2)
        custom_params = data.get('custom_params', {})
        experiment_name = data.get('experiment_name')
        
        config_manager = get_config_manager()
        config = config_manager.create_default_config(algorithm, n_agents)
        
        if experiment_name:
            config.experiment_name = experiment_name
        
        # Apply custom parameters
        for param_path, value in custom_params.items():
            config_manager._set_nested_param(config, param_path, value)
        
        # Save configuration
        config_path = config_manager.save_config(config)
        
        return jsonify({
            'success': True,
            'config_path': config_path,
            'experiment_name': config.experiment_name,
            'config_summary': {
                'algorithm': config.algorithm,
                'n_agents': len(config.agents),
                'shared_memory': any(agent.memory.shared_memory for agent in config.agents),
                'communication_enabled': any(agent.communication.enabled for agent in config.agents),
                'centralized_training': any(agent.centralized_training for agent in config.agents),
                'parameter_sharing': config.agents[0].policy_network.parameter_sharing if config.agents else 'none'
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/config/memory-settings', methods=['POST'])
def configure_memory_settings():
    """Configure memory settings for agents."""
    try:
        data = request.json
        config_name = data.get('config_name')
        memory_settings = data.get('memory_settings')
        
        config_manager = get_config_manager()
        config = config_manager.load_config(f"config/experiments/{config_name}.yaml")
        
        # Apply memory settings
        for agent_id, settings in memory_settings.items():
            if agent_id < len(config.agents):
                agent = config.agents[agent_id]
                for key, value in settings.items():
                    if hasattr(agent.memory, key):
                        setattr(agent.memory, key, value)
        
        # Save updated configuration
        config_path = config_manager.save_config(config)
        
        return jsonify({
            'success': True,
            'config_path': config_path,
            'memory_summary': {
                f'agent_{i}': {
                    'memory_type': agent.memory.memory_type,
                    'shared_memory': agent.memory.shared_memory,
                    'memory_size': agent.memory.memory_size,
                    'experience_sharing': agent.memory.experience_sharing
                } for i, agent in enumerate(config.agents)
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/config/individual-agent/<int:agent_id>', methods=['POST'])
def configure_individual_agent(agent_id):
    """Configure parameters for an individual agent."""
    try:
        data = request.json
        config_name = data.get('config_name')
        agent_settings = data.get('agent_settings')
        
        config_manager = get_config_manager()
        config = config_manager.load_config(f"config/experiments/{config_name}.yaml")
        
        if agent_id >= len(config.agents):
            return jsonify({'success': False, 'error': 'Agent ID out of range'}), 400
        
        agent = config.agents[agent_id]
        
        # Apply agent-specific settings
        for key, value in agent_settings.items():
            if hasattr(agent, key):
                setattr(agent, key, value)
            elif key in ['memory_settings']:
                for mem_key, mem_value in value.items():
                    if hasattr(agent.memory, mem_key):
                        setattr(agent.memory, mem_key, mem_value)
            elif key in ['network_settings']:
                for net_key, net_value in value.items():
                    if hasattr(agent.policy_network, net_key):
                        setattr(agent.policy_network, net_key, net_value)
            elif key in ['communication_settings']:
                for comm_key, comm_value in value.items():
                    if hasattr(agent.communication, comm_key):
                        setattr(agent.communication, comm_key, comm_value)
        
        # Save updated configuration
        config_path = config_manager.save_config(config)
        
        return jsonify({
            'success': True,
            'config_path': config_path,
            'agent_summary': {
                'agent_id': agent.agent_id,
                'learning_rate': agent.learning_rate,
                'gamma': agent.gamma,
                'memory_shared': agent.memory.shared_memory,
                'communication_enabled': agent.communication.enabled,
                'network_architecture': agent.policy_network.hidden_dims
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    # Detect if running in GitHub Codespaces
    CODESPACE_NAME = os.environ.get('CODESPACE_NAME')
    
    if CODESPACE_NAME:
        # Update CORS origins for Codespaces
        frontend_url = f"https://{CODESPACE_NAME}-3000.app.github.dev"
        backend_url = f"https://{CODESPACE_NAME}-5000.app.github.dev"
        
        # Update CORS configuration
        app.config['CORS_ORIGINS'] = [
            frontend_url,
            "http://localhost:3000",
            "https://shreyanmitra.github.io"
        ]
        
        print("🚀 Starting EasyMARL Flask Backend in GitHub Codespaces")
        print("=" * 60)
        print("🎓 GitHub Student Pack - Free ML Development Environment")
        print(f"📍 Codespace: {CODESPACE_NAME}")
        print(f"📱 Frontend URL: {frontend_url}")
        print(f"🔌 Backend URL:  {backend_url}")
        print("🔗 API available at /api")
        print("🔬 Research API available at /api/research")
        print("=" * 60)
        
    else:
        print("🚀 Starting EasyMARL Flask Backend")
        print("=" * 60)
        print("🧠 Multi-Agent Reinforcement Learning Framework")
        print("📊 Research-Grade Algorithm Discovery & Experimentation")
        print("🔬 Advanced Hyperparameter Optimization")
        print("📱 React frontend should be built and available")
        print("🔗 API available at http://localhost:5000/api")
        print("🌐 Full app available at http://localhost:5000")
        print("🔬 Research API available at http://localhost:5000/api/research")
    
    # Set port from environment or default to 5000
    port = int(os.environ.get('PORT', 5000))
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=True,
        threaded=True
    )
