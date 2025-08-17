"""
Model-Based Reinforcement Learning Algorithms

This package contains algorithms that use models of the environment
to plan actions and improve learning efficiency.

Categories:
- given_model: Use predefined environment models (e.g., board games)
- learned_model: Learn environment models from data
"""

from .given_model import *
from .learned_model import *

__all__ = [
    # Model-based algorithms will be added as they are implemented
]
