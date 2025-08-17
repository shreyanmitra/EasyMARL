"""
Value-Based Reinforcement Learning Algorithms

These algorithms learn value functions that estimate the expected
return from states or state-action pairs.

Categories:
- approximation: Use function approximation (neural networks)
- tabular: Use lookup tables for value storage
"""

from .approximation import *
from .tabular import *

__all__ = [
    # Value-based algorithms
    'qmix', 'vdn', 'qtran', 'iql', 'mfq',  # Approximation
    'nashq', 'minimaxq', 'wolfphc', 'hql', 'lql'  # Tabular
]
