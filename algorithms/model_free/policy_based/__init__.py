"""
Policy-Based Reinforcement Learning Algorithms

These algorithms directly learn policies that map states to actions,
optimizing them using gradient-based methods.

Categories:
- discrete_action: For environments with discrete action spaces
- continuous_action: For environments with continuous action spaces
"""

from .discrete_action import *
from .continuous_action import *

__all__ = [
    # Policy-based algorithms
    'ippo', 'mappo',  # Discrete action
    'maddpg', 'maddpgcomm'  # Continuous action
]
