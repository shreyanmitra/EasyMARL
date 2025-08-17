"""
Function Approximation Value-Based Algorithms

These algorithms use neural networks or other function approximators
to represent value functions, enabling scalability to large state spaces.

Algorithms:
- QMIX: Monotonic value decomposition for cooperative multi-agent tasks
- VDN: Simple additive value decomposition  
- QTRAN: General value decomposition with additional constraints
- IQL: Independent Q-Learning with function approximation
- MFQ: Mean Field Q-Learning for large-scale multi-agent systems
"""

# Import all algorithms
from .qmix import QMIXAlgorithm
from .vdn import VDNAlgorithm  
from .qtran import QTRANAlgorithm
from .iql import IQLAlgorithm
from .mfq import MFQAlgorithm

__all__ = ['QMIXAlgorithm', 'VDNAlgorithm', 'QTRANAlgorithm', 'IQLAlgorithm', 'MFQAlgorithm']
