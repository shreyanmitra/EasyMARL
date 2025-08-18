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

# Export key functions
__all__ = [
    'make_env',
    'train_agents',
    'get_config_manager',
    'SimpleMultiAgentController',
    'ModernMultiAgentController'
]
