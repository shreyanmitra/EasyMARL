"""
Actor-Critic Reinforcement Learning Algorithms

These algorithms combine value-based and policy-based approaches,
using both a value function (critic) and a policy (actor).

Algorithms:
- COMA: Counterfactual Multi-Agent Policy Gradients
- COMA-Comm: COMA with communication
- MAACC: Multi-Agent Actor-Critic with Communication
- MAVEN: Multi-Agent Variational Exploration
- DCG: Deep Coordination Graphs  
- NFSP: Neural Fictitious Self-Play
- MADDPG: Multi-Agent Deep Deterministic Policy Gradient (moved from policy_based)
- MADDPG-Comm: MADDPG with communication (moved from policy_based)
"""

# Import all algorithms
from .coma import COMAAlgorithm
from .comacomm import COMACommAlgorithm
from .maacc import MAACCAlgorithm
from .maven import MAVENAlgorithm
from .dcg import DCGAlgorithm
from .nfsp import NFSPAlgorithm
from .maddpg import MADDPG
from .maddpgcomm import MADDPGComm

__all__ = [
    'COMAAlgorithm', 'COMACommAlgorithm', 'MAACCAlgorithm', 'MAVENAlgorithm', 
    'DCGAlgorithm', 'NFSPAlgorithm', 'MADDPG', 'MADDPGComm'
]
