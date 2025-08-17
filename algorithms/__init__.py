"""
Algorithm factory for creating and managing MARL algorithms.

This module provides a unified interface for creating and configuring
different multi-agent reinforcement learning algorithms.
"""

from typing import Dict, Type
import torch

from .base import MARLAlgorithm
from .ippo import IPPO
from .maddpg import MADDPG
from .qmix import QMIX
from .mappo import MAPPO
from .iql import IQL
from .vdn import VDN
from .coma import COMA
from .qtran import QTRAN
from .maven import MAVEN
from .hql import HQL
from .lql import LQL
from .wolfphc import WoLFPHC
from .nashq import NashQ
from .dcg import DCG
from .minimaxq import MinimaxQ
from .maacc import MAACC
from .nfsp import NFSP
from .mfq import MFQ
from .maddpgcomm import MADDPGComm
from .comacomm import COMAComm


class AlgorithmFactory:
    """
    Factory class for creating MARL algorithms.
    """
    
    # Registry of available algorithms
    ALGORITHMS = {
        'ippo': IPPO,
        'maddpg': MADDPG,
        'qmix': QMIX,
        'mappo': MAPPO,
        'iql': IQL,
        'vdn': VDN,
        'coma': COMA,
        'qtran': QTRAN,
        'maven': MAVEN,
        'hql': HQL,
        'lql': LQL,
        'wolfphc': WoLFPHC,
        'nashq': NashQ,
        'dcg': DCG,
        'minimaxq': MinimaxQ,
        'maacc': MAACC,
        'nfsp': NFSP,
        'mfq': MFQ,
        'maddpgcomm': MADDPGComm,
        'comacomm': COMAComm,
        'ppo': IPPO,  # Alias for backward compatibility
    }
    
    @classmethod
    def create_algorithm(cls, algorithm_name: str, env, config: Dict, device: torch.device) -> MARLAlgorithm:
        """
        Create a MARL algorithm instance.
        
        Args:
            algorithm_name: Name of the algorithm to create
            env: Multi-agent environment
            config: Configuration dictionary
            device: PyTorch device
            
        Returns:
            Initialized MARL algorithm instance
            
        Raises:
            ValueError: If the algorithm name is not recognized
        """
        algorithm_name = algorithm_name.lower()
        
        if algorithm_name not in cls.ALGORITHMS:
            available_algorithms = list(cls.ALGORITHMS.keys())
            raise ValueError(
                f"Unknown algorithm '{algorithm_name}'. "
                f"Available algorithms: {available_algorithms}"
            )
        
        algorithm_class = cls.ALGORITHMS[algorithm_name]
        
        # Add algorithm-specific configuration
        config = cls._prepare_config(algorithm_name, config)
        
        print(f"Creating {algorithm_name.upper()} algorithm...")
        return algorithm_class(env, config, device)
    
    @classmethod
    def _prepare_config(cls, algorithm_name: str, config: Dict) -> Dict:
        """
        Prepare algorithm-specific configuration.
        
        Args:
            algorithm_name: Name of the algorithm
            config: Base configuration dictionary
            
        Returns:
            Updated configuration dictionary
        """
        config = config.copy()  # Don't modify original config
        
        # Set algorithm-specific defaults
        if algorithm_name in ['ippo', 'ppo']:
            cls._set_ippo_defaults(config)
        elif algorithm_name == 'maddpg':
            cls._set_maddpg_defaults(config)
        elif algorithm_name == 'qmix':
            cls._set_qmix_defaults(config)
        elif algorithm_name == 'mappo':
            cls._set_mappo_defaults(config)
        
        return config
    
    @classmethod
    def _set_ippo_defaults(cls, config: Dict):
        """Set IPPO-specific default values."""
        defaults = {
            'gamma': 0.99,
            'lambda_gae': 0.95,
            'clip_epsilon': 0.2,
            'value_loss_coef': 0.5,
            'entropy_coef': 0.01,
            'max_grad_norm': 0.5,
            'ppo_epochs': 4,
            'mini_batch_size': 64,
            'rollout_length': 128,
            'lr': 3e-4,
            'lr_decay_steps': 1000,
            'lr_decay': 0.99,
            'hidden_dim': 128
        }
        
        for key, value in defaults.items():
            if key not in config:
                config[key] = value
    
    @classmethod
    def _set_maddpg_defaults(cls, config: Dict):
        """Set MADDPG-specific default values."""
        defaults = {
            'gamma': 0.99,
            'tau': 0.01,
            'exploration_noise': 0.1,
            'lr_actor': 1e-4,
            'lr_critic': 1e-3,
            'temperature': 1.0,
            'hard_gumbel': True,
            'batch_size': 64,
            'buffer_size': 100000,
            'learning_starts': 1000,
            'update_freq': 1,
            'target_update_freq': 1,
            'hidden_dim': 128
        }
        
        for key, value in defaults.items():
            if key not in config:
                config[key] = value
    
    @classmethod
    def _set_qmix_defaults(cls, config: Dict):
        """Set QMIX-specific default values."""
        defaults = {
            'gamma': 0.99,
            'epsilon_start': 1.0,
            'epsilon_end': 0.05,
            'epsilon_decay': 0.995,
            'batch_size': 32,
            'buffer_size': 50000,
            'learning_starts': 1000,
            'train_freq': 4,
            'target_update_freq': 200,
            'lr': 5e-4,
            'grad_norm_clip': 10.0,
            'mixing_embed_dim': 32,
            'hypernet_embed_dim': 64,
            'hidden_dim': 128,
            'max_episode_length': 100
        }
        
        for key, value in defaults.items():
            if key not in config:
                config[key] = value
    
    @classmethod
    def _set_mappo_defaults(cls, config: Dict):
        """Set MAPPO-specific default values."""
        defaults = {
            'gamma': 0.99,
            'lambda_gae': 0.95,
            'clip_epsilon': 0.2,
            'value_loss_coef': 0.5,
            'entropy_coef': 0.01,
            'max_grad_norm': 0.5,
            'ppo_epochs': 4,
            'mini_batch_size': 32,
            'rollout_length': 128,
            'lr_actor': 3e-4,
            'lr_critic': 1e-3,
            'share_parameters': False,
            'hidden_dim': 128
        }
        
        for key, value in defaults.items():
            if key not in config:
                config[key] = value
    
    @classmethod
    def get_available_algorithms(cls) -> List[str]:
        """Get list of available algorithm names."""
        return list(cls.ALGORITHMS.keys())
    
    @classmethod
    def get_algorithm_info(cls, algorithm_name: str) -> Dict:
        """
        Get information about a specific algorithm.
        
        Args:
            algorithm_name: Name of the algorithm
            
        Returns:
            Dictionary containing algorithm information
        """
        algorithm_name = algorithm_name.lower()
        
        info = {
            'ippo': {
                'name': 'Independent Proximal Policy Optimization',
                'type': 'On-policy',
                'description': 'Each agent learns independently using PPO with GAE',
                'best_for': 'Fully cooperative tasks, simple coordination'
            },
            'maddpg': {
                'name': 'Multi-Agent Deep Deterministic Policy Gradient',
                'type': 'Off-policy',
                'description': 'Centralized training with decentralized execution using DDPG',
                'best_for': 'Mixed-motive scenarios, continuous action spaces'
            },
            'qmix': {
                'name': 'Q-learning with Mixing Networks',
                'type': 'Off-policy',
                'description': 'Value-based method with mixing network for joint action-values',
                'best_for': 'Fully cooperative tasks, discrete actions'
            },
            'mappo': {
                'name': 'Multi-Agent Proximal Policy Optimization',
                'type': 'On-policy',
                'description': 'PPO with centralized value functions and optional parameter sharing',
                'best_for': 'Fully cooperative tasks, complex coordination'
            }
        }
        
        return info.get(algorithm_name, {'name': 'Unknown', 'type': 'Unknown', 'description': 'No information available'})


def create_marl_algorithm(algorithm_name: str, env, config: Dict, device: torch.device) -> MARLAlgorithm:
    """
    Convenience function to create a MARL algorithm.
    
    Args:
        algorithm_name: Name of the algorithm to create
        env: Multi-agent environment
        config: Configuration dictionary
        device: PyTorch device
        
    Returns:
        Initialized MARL algorithm instance
    """
    return AlgorithmFactory.create_algorithm(algorithm_name, env, config, device)


def list_available_algorithms() -> None:
    """Print information about all available algorithms."""
    algorithms = AlgorithmFactory.get_available_algorithms()
    
    print("Available MARL Algorithms:")
    print("=" * 50)
    
    for alg_name in algorithms:
        if alg_name == 'ppo':  # Skip alias
            continue
            
        info = AlgorithmFactory.get_algorithm_info(alg_name)
        print(f"\n{alg_name.upper()}: {info['name']}")
        print(f"Type: {info['type']}")
        print(f"Description: {info['description']}")
        print(f"Best for: {info['best_for']}")
    
    print("\n" + "=" * 50)


if __name__ == "__main__":
    # Demo: list all available algorithms
    list_available_algorithms()
