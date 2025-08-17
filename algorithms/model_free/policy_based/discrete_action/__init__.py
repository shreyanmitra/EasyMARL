"""
Discrete Action Policy-Based Algorithms

These algorithms learn policies for environments with discrete,
countable action spaces using policy gradient methods.

Algorithms:
- IPPO: Independent Proximal Policy Optimization
- MAPPO: Multi-Agent Proximal Policy Optimization with centralized training
"""

# Import all algorithms
from .ippo import IPPOAlgorithm
from .mappo import MAPPOAlgorithm

__all__ = ['IPPOAlgorithm', 'MAPPOAlgorithm']
