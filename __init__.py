"""
EasyMARL - Educational Multi-Agent Reinforcement Learning Framework

A comprehensive, beginner-friendly framework for multi-agent reinforcement learning
that provides both educational simplicity and production-ready performance.

Key Features:
🎓 Educational Focus: Perfect for learning MARL concepts
🚀 20+ Algorithms: Comprehensive algorithm implementations  
🎮 Interactive GUI: Web-based training interface
⚡ World-Class Performance: 10x faster with enhanced features
📊 Rich Visualizations: Real-time training monitoring
🔬 Research Ready: Professional experiment tracking

Quick Start:
    >>> import easymarl
    >>> from easymarl import SimpleMultiAgentController
    >>> 
    >>> # Create environment and controller
    >>> env = easymarl.make_env("MultiGrid-Empty-6x6-v0")
    >>> controller = SimpleMultiAgentController(
    ...     env=env, 
    ...     algorithm="qmix",
    ...     config=easymarl.get_default_config("qmix")
    ... )
    >>> 
    >>> # Train agents
    >>> controller.train(episodes=1000)
    >>> 
    >>> # Evaluate performance
    >>> results = controller.evaluate()

Web GUI:
    >>> easymarl.launch_gui()  # Opens web interface
    
    Or from command line:
    $ easymarl-gui

For detailed documentation and tutorials, visit:
https://github.com/shreyanmitra/EasyMARL
"""

__version__ = "1.0.0"
__author__ = "Shreyan Mitra"
__email__ = "shreyan.m.mitra@gmail.com"
__license__ = "MIT"

# Core imports for easy access
from .utils import (
    make_env, 
    get_default_config,
    ENHANCED_FEATURES_AVAILABLE,
    ADVANCED_FEATURES_AVAILABLE
)

# Controller imports
from .src.controllers.simple_multiagent_controller import SimpleMultiAgentController
from .src.controllers.modern_multiagent_controller import ModernMultiAgentController  
from .src.controllers.vectorized_controller import VectorizedController

# Algorithm factory
from .algorithms import create_marl_algorithm, list_available_algorithms

# GUI launcher
def launch_gui():
    """Launch the EasyMARL web-based GUI interface."""
    try:
        from .gui import main as gui_main
        gui_main()
    except ImportError:
        print("GUI dependencies not installed. Install with: pip install easymarl[gui]")
    except Exception as e:
        print(f"Error launching GUI: {e}")
        print("Make sure you have installed GUI dependencies: pip install easymarl[gui]")

# Configuration management
from .src.core.config_manager import ConfigManager

# Public API
__all__ = [
    # Core utilities
    "make_env",
    "get_default_config", 
    "ENHANCED_FEATURES_AVAILABLE",
    "ADVANCED_FEATURES_AVAILABLE",
    
    # Controllers
    "SimpleMultiAgentController",
    "ModernMultiAgentController", 
    "VectorizedController",
    
    # Algorithm management
    "create_marl_algorithm",
    "list_available_algorithms",
    
    # GUI
    "launch_gui",
    
    # Configuration
    "ConfigManager",
    
    # Version info
    "__version__",
    "__author__",
    "__email__",
    "__license__"
]

# Available algorithms
ALGORITHMS = [
    # Value-based methods
    "qmix", "vdn", "iql", "qtran",
    
    # Policy-based methods  
    "ippo", "mappo", "maddpg", "maddpgcomm",
    
    # Actor-critic methods
    "coma", "comacomm", "maacc", "dcg", "maven",
    
    # Game-theoretic methods
    "nfsp", "minimaxq", "wolfphc",
    
    # Hierarchical methods
    "hql", "lql",
    
    # Mean field methods
    "mfq"
]

# Supported environments
ENVIRONMENTS = [
    "MultiGrid-Empty-6x6-v0",
    "MultiGrid-Empty-8x8-v0", 
    "MultiGrid-Empty-16x16-v0",
    "MultiGrid-FourRooms-v0",
    "MultiGrid-DoorKey-6x6-v0",
    "MultiGrid-DoorKey-8x8-v0",
    "MultiGrid-Cluttered-6x6-v0",
    "MultiGrid-Cluttered-8x8-v0",
    "MultiGrid-Maze-6x6-v0",
    "MultiGrid-Maze-8x8-v0",
    "MultiGrid-CoinGame-v0",
    "MultiGrid-Gather-v0"
]

def print_info():
    """Print EasyMARL information and available features."""
    print(f"🎓 EasyMARL v{__version__}")
    print("Educational Multi-Agent Reinforcement Learning Framework")
    print(f"📚 {len(ALGORITHMS)} algorithms available")
    print(f"🎮 {len(ENVIRONMENTS)} environments supported")
    print(f"⚡ Enhanced features: {'✅' if ENHANCED_FEATURES_AVAILABLE else '❌'}")
    print(f"🔬 Advanced tracking: {'✅' if ADVANCED_FEATURES_AVAILABLE else '❌'}")
    print("\n🚀 Quick start: easymarl.launch_gui()")
    print("📖 Documentation: https://github.com/shreyanmitra/EasyMARL")

def get_info():
    """Get EasyMARL information as a dictionary."""
    return {
        "version": __version__,
        "algorithms": ALGORITHMS,
        "environments": ENVIRONMENTS,
        "enhanced_features": ENHANCED_FEATURES_AVAILABLE,
        "advanced_features": ADVANCED_FEATURES_AVAILABLE,
        "author": __author__,
        "license": __license__
    }

# Convenience function for users
info = print_info
