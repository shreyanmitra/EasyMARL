"""
Tabular Value-Based Algorithms

These algorithms use lookup tables to store exact value estimates,
suitable for environments with small, discrete state spaces.

Algorithms:
- Nash-Q: Game-theoretic equilibrium-based learning
- Minimax-Q: Competitive multi-agent learning with minimax strategy
- WoLF-PHC: Win-or-Learn-Fast with Policy Hill Climbing
- HQL: Hysteretic Q-Learning for non-stationary environments
- LQL: Lenient Q-Learning for cooperative environments
"""

# Import all algorithms
from .nashq import NashQAlgorithm
from .minimaxq import MinimaxQAlgorithm
from .wolfphc import WoLFPHCAlgorithm
from .hql import HQLAlgorithm
from .lql import LQLAlgorithm

__all__ = ['NashQAlgorithm', 'MinimaxQAlgorithm', 'WoLFPHCAlgorithm', 'HQLAlgorithm', 'LQLAlgorithm']
