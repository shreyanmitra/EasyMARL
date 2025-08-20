"""
EasyMARL: Educational Multi-Agent Reinforcement Learning Framework

A comprehensive, educational framework for Multi-Agent Reinforcement Learning (MARL)
designed to make MARL accessible to beginners while providing advanced features for researchers.

Author: Shreyan Mitra
Email: shreyan.m.mitra@gmail.com
GitHub: https://github.com/shreyanmitra/EasyMARL

Key Components:
- Core utilities and environment management
- 20+ MARL algorithms across all major categories
- Modern web-based GUI with React frontend
- Advanced vectorization and performance optimization
- Educational documentation and examples
"""

__version__ = "1.0.0"
__author__ = "Shreyan Mitra"
__email__ = "shreyan.m.mitra@gmail.com"

# Import main utilities for backward compatibility
from .core.utils import *
from .core.config_manager import *

# Import main components
from .algorithms import *
from .controllers import *
from .environments import *

# GUI launcher function
def launch_gui(port=7860, share=False, educational_mode=True):
    """
    Launch the Gradio web interface for EasyMARL.
    
    Args:
        port (int): Port to run the interface on (default: 7860)
        share (bool): Whether to create a shareable public link (default: False)
        educational_mode (bool): Enable educational explanations (default: True)
    
    Returns:
        None: Launches the web interface in the default browser
    """
    try:
        from .gui.gradio_interface import main as launch_gradio
        print("🚀 Launching EasyMARL Gradio Interface...")
        print(f"📡 Running on: http://localhost:{port}")
        if educational_mode:
            print("🎓 Educational mode enabled - detailed explanations included")
        launch_gradio(port=port, share=share, educational_mode=educational_mode)
    except ImportError as e:
        print("❌ Gradio interface not available. Install with: pip install easymarl[gui]")
        print(f"Error: {e}")

def make_env(env_name, **kwargs):
    """Create and return a multi-agent environment."""
    from .environments import VectorizedEnv
    return VectorizedEnv(env_name, **kwargs)

def get_default_config(algorithm_name):
    """Get default configuration for a specific algorithm."""
    from .core.config_manager import ConfigManager
    return ConfigManager().get_algorithm_config(algorithm_name)

# Export key functions
__all__ = [
    'launch_gui',
    'make_env', 
    'get_default_config',
    'train_agents',
    'get_config_manager',
    'UnifiedMultiAgentController',
    'SimpleMultiAgentController',
    'ModernMultiAgentController'
]
