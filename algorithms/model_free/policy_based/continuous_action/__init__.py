"""
Continuous Action Policy-Based Algorithms

These algorithms learn policies for environments with continuous
action spaces, typically using deterministic or Gaussian policies.

Algorithms:
- MADDPG: Multi-Agent Deep Deterministic Policy Gradient
- MADDPG-Comm: MADDPG with communication capabilities
"""

# Import all algorithms
from .maddpg import MADDPGAlgorithm
from .maddpgcomm import MADDPGCommAlgorithm

__all__ = ['MADDPGAlgorithm', 'MADDPGCommAlgorithm']
