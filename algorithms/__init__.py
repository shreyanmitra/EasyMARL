"""
Multi-Agent Reinforcement Learning Algorithm Library

This package provides implementations of various multi-agent reinforcement learning
algorithms, organized using a comprehensive taxonomy for easy understanding and use.

Taxonomy Structure:
📁 Model-Free (Learn directly from experience)
  ├── 📁 Value-Based (Learn value functions)
  │   ├── 📁 Approximation (Neural networks): QMIX, VDN, QTRAN, IQL, MFQ
  │   └── 📁 Tabular (Lookup tables): Nash-Q, Minimax-Q, WoLF-PHC, HQL, LQL
  ├── 📁 Policy-Based (Learn policies directly)
  │   ├── 📁 Discrete Action: IPPO, MAPPO
  │   └── 📁 Continuous Action: MADDPG, MADDPG-Comm
  └── 📁 Actor-Critic (Combine value and policy): COMA, COMA-Comm, MAACC, MAVEN, DCG, NFSP

📁 Model-Based (Use environment models)
  ├── 📁 Given Model (Predefined dynamics): Future algorithms
  └── 📁 Learned Model (Learn dynamics): Future algorithms

Each algorithm follows the MARLAlgorithm base class interface for consistency.
"""

from typing import Dict, Type, List
import torch

# Import base classes
from .base import MARLAlgorithm
from .taxonomy import taxonomy, AlgorithmCategory, ModelFreeCategory

# Import all algorithms from their taxonomical locations
from .model_free.value_based.approximation.qmix import QMIX
from .model_free.value_based.approximation.vdn import VDN
from .model_free.value_based.approximation.qtran import QTRAN
from .model_free.value_based.approximation.iql import IQL
from .model_free.value_based.approximation.mfq import MFQ

from .model_free.value_based.tabular.nashq import NashQ
from .model_free.value_based.tabular.minimaxq import MinimaxQ
from .model_free.value_based.tabular.wolfphc import WoLFPHC
from .model_free.value_based.tabular.hql import HQL
from .model_free.value_based.tabular.lql import LQL

from .model_free.policy_based.discrete_action.ippo import IPPO
from .model_free.policy_based.discrete_action.mappo import MAPPO

from .model_free.policy_based.continuous_action.maddpg import MADDPG
from .model_free.policy_based.continuous_action.maddpgcomm import MADDPGComm

from .model_free.actor_critic.coma import COMA
from .model_free.actor_critic.comacomm import COMAComm
from .model_free.actor_critic.maacc import MAACC
from .model_free.actor_critic.maven import MAVEN
from .model_free.actor_critic.dcg import DCG
from .model_free.actor_critic.nfsp import NFSP


class AlgorithmFactory:
    """
    Taxonomy-aware factory class for creating MARL algorithms.
    
    This factory uses the comprehensive RL taxonomy to organize algorithms
    and provide educational guidance on algorithm selection.
    
    Educational Features:
    - Categorizes algorithms by their fundamental principles
    - Provides learning progression recommendations
    - Offers algorithm selection guidance based on requirements
    
    Research Features:
    - Systematic comparison within categories
    - Clear identification of algorithm relationships
    - Structured evaluation framework
    """
    
    # Taxonomy-organized algorithm registry
    ALGORITHMS = {
        # Value-Based Approximation
        'qmix': QMIX,
        'vdn': VDN,
        'qtran': QTRAN,
        'iql': IQL,
        'mfq': MFQ,
        
        # Value-Based Tabular
        'nashq': NashQ,
        'minimaxq': MinimaxQ,
        'wolfphc': WoLFPHC,
        'hql': HQL,
        'lql': LQL,
        
        # Policy-Based Discrete Action
        'ippo': IPPO,
        'mappo': MAPPO,
        
        # Policy-Based Continuous Action
        'maddpg': MADDPG,
        'maddpgcomm': MADDPGComm,
        
        # Actor-Critic
        'coma': COMA,
        'comacomm': COMAComm,
        'maacc': MAACC,
        'maven': MAVEN,
        'dcg': DCG,
        'nfsp': NFSP,
        
        # Backward compatibility aliases
        'ppo': IPPO,
    }
    
    @classmethod
    def get_algorithms_by_category(cls, category: str) -> Dict[str, Type[MARLAlgorithm]]:
        """
        Get algorithms organized by taxonomical category.
        
        Args:
            category: Category name ('value_based', 'policy_based', 'actor_critic', etc.)
            
        Returns:
            Dict mapping algorithm names to classes within the category
        """
        category_mapping = {
            'value_based_approximation': ['qmix', 'vdn', 'qtran', 'iql', 'mfq'],
            'value_based_tabular': ['nashq', 'minimaxq', 'wolfphc', 'hql', 'lql'],
            'policy_based_discrete': ['ippo', 'mappo'],
            'policy_based_continuous': ['maddpg', 'maddpgcomm'],
            'actor_critic': ['coma', 'comacomm', 'maacc', 'maven', 'dcg', 'nfsp'],
            
            # Broader categories
            'value_based': ['qmix', 'vdn', 'qtran', 'iql', 'mfq', 'nashq', 'minimaxq', 'wolfphc', 'hql', 'lql'],
            'policy_based': ['ippo', 'mappo', 'maddpg', 'maddpgcomm'],
            'beginner': ['vdn', 'ippo', 'nashq'],
            'intermediate': ['qmix', 'mappo', 'coma'],
            'advanced': ['qtran', 'maven', 'maacc'],
            'cooperative': ['qmix', 'vdn', 'ippo', 'mappo', 'coma'],
            'competitive': ['nashq', 'minimaxq', 'wolfphc', 'nfsp'],
        }
        
        algorithm_names = category_mapping.get(category, [])
        return {name: cls.ALGORITHMS[name] for name in algorithm_names if name in cls.ALGORITHMS}
    
    @classmethod
    def get_recommended_algorithms(cls, **criteria) -> List[str]:
        """
        Get algorithm recommendations based on user criteria.
        
        Keyword Args:
            environment_type: 'cooperative', 'competitive', 'mixed'
            experience_level: 'beginner', 'intermediate', 'advanced'
            action_space: 'discrete', 'continuous'
            state_space: 'small', 'large'
            
        Returns:
            List of recommended algorithm names
        """
        return taxonomy.get_algorithm_recommendations(**criteria)
    
    @classmethod
    def get_learning_progression(cls) -> List[Dict]:
        """Get educational learning progression through algorithms."""
        return taxonomy.get_learning_progression()
    
    @classmethod
    def get_algorithm_path(cls, algorithm_name: str) -> str:
        """Get the taxonomical path for an algorithm."""
        return taxonomy.get_algorithm_path(algorithm_name) or "Unknown"
    
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
