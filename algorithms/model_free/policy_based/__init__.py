"""
Policy-Based Reinforcement Learning Algorithms

These algorithms directly learn policies that map states to actions,
optimizing them using gradient-based methods.

Categories:
- discrete_action: For environments with discrete action spaces (like MultiGrid)

Note: MADDPG has been moved to actor_critic since it's an actor-critic algorithm,
not a pure policy-based method.
"""

from .discrete_action import *

__all__ = [
    # Policy-based algorithms  
    'ippo', 'mappo',  # Discrete action algorithms
]
