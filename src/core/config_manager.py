"""
Configuration Management System for EasyMARL Research Platform

This module provides utilities for managing experimental configurations,
hyperparameter sets, and research templates for systematic MARL experimentation.
"""

import os
import yaml
import json
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict, field
from pathlib import Path
import datetime
from copy import deepcopy


@dataclass
class MemoryConfig:
    """Configuration for agent memory/replay buffer."""
    memory_type: str = "replay_buffer"  # replay_buffer, prioritized_replay, shared_memory
    memory_size: int = 10000
    shared_memory: bool = False  # Whether memory is shared between agents
    shared_memory_ratio: float = 0.7  # Ratio of shared vs individual memory
    
    # Prioritized replay parameters
    use_prioritized: bool = False
    alpha: float = 0.6  # Prioritization exponent
    beta: float = 0.4  # Importance sampling exponent
    beta_schedule: str = "linear"  # linear, constant, exponential
    
    # Memory management
    min_memory_size: int = 1000  # Minimum size before training
    memory_warmup: int = 1000  # Steps before using memory
    sample_efficiency: str = "standard"  # standard, importance_sampling, hindsight
    
    # Experience sharing
    experience_sharing: bool = False
    sharing_frequency: int = 100  # Steps between sharing
    sharing_method: str = "random"  # random, best_performance, diverse


@dataclass  
class NetworkConfig:
    """Configuration for neural networks."""
    # Architecture
    hidden_dims: List[int] = field(default_factory=lambda: [64, 64])
    activation: str = "relu"  # relu, tanh, gelu, swish
    output_activation: Optional[str] = None
    
    # Regularization
    dropout: float = 0.0
    batch_norm: bool = False
    layer_norm: bool = False
    weight_decay: float = 0.0
    
    # Initialization
    weight_init: str = "xavier_uniform"  # xavier_uniform, xavier_normal, kaiming_uniform, kaiming_normal, orthogonal
    bias_init: str = "zeros"  # zeros, constant
    
    # Network sharing
    shared_networks: bool = False  # Share networks between agents
    parameter_sharing: str = "none"  # none, full, partial, selective
    
    # Specialized networks
    use_dueling: bool = False  # For DQN variants
    use_noisy_nets: bool = False  # Noisy networks for exploration
    use_attention: bool = False  # Attention mechanisms
    attention_heads: int = 4
    
    # Network updates
    soft_update: bool = True
    tau: float = 0.005  # Soft update rate
    target_update_freq: int = 100  # Hard update frequency
    
    # Optimizer settings
    optimizer: str = "adam"  # adam, rmsprop, sgd, adamw
    lr_schedule: str = "constant"  # constant, linear_decay, exponential_decay, cosine
    gradient_clipping: Optional[float] = None
    
    # Advanced features
    spectral_norm: bool = False
    residual_connections: bool = False
    multi_head_output: bool = False


@dataclass
class CommunicationConfig:
    """Configuration for agent communication."""
    enabled: bool = False
    communication_type: str = "direct"  # direct, broadcast, selective, hierarchical
    
    # Communication channels
    message_size: int = 32
    max_messages: int = 5
    communication_range: Optional[float] = None  # None for unlimited
    
    # Communication learning
    learnable_communication: bool = True
    communication_loss_weight: float = 0.1
    differentiable: bool = True
    
    # Protocol
    protocol: str = "emergent"  # emergent, predefined, compositional
    vocabulary_size: int = 100
    message_encoding: str = "continuous"  # continuous, discrete, symbolic
    
    # Coordination
    centralized_communication: bool = False
    communication_scheduler: str = "always"  # always, on_demand, scheduled
    bandwidth_limit: Optional[int] = None


@dataclass
class AgentConfig:
    """Comprehensive configuration for a single agent."""
    agent_id: int
    agent_type: str = "learner"  # learner, scripted, human, random
    
    # Learning parameters
    learning_rate: float = 0.001
    gamma: float = 0.99  # Discount factor
    epsilon: float = 0.1  # Exploration rate
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    epsilon_schedule: str = "exponential"  # exponential, linear, step, cosine
    
    # Training parameters
    batch_size: int = 32
    update_frequency: int = 1  # Steps between updates
    train_frequency: int = 4  # Experience replay frequency
    
    # Algorithm-specific parameters
    # PPO/Actor-Critic
    clip_ratio: Optional[float] = None
    entropy_coef: Optional[float] = None
    value_coef: Optional[float] = None
    gae_lambda: Optional[float] = None
    ppo_epochs: Optional[int] = None
    
    # DQN variants
    double_dqn: bool = False
    dueling_dqn: bool = False
    n_step: int = 1
    
    # MADDPG/Continuous control
    actor_lr: Optional[float] = None
    critic_lr: Optional[float] = None
    exploration_noise: Optional[float] = None
    action_noise_std: Optional[float] = None
    noise_clip: Optional[float] = None
    
    # Multi-agent specific
    centralized_training: bool = True
    decentralized_execution: bool = True
    observation_sharing: bool = False
    action_sharing: bool = False
    reward_sharing: bool = False
    
    # Memory configuration
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    
    # Network configuration
    policy_network: NetworkConfig = field(default_factory=NetworkConfig)
    value_network: Optional[NetworkConfig] = None
    critic_network: Optional[NetworkConfig] = None
    
    # Communication configuration
    communication: CommunicationConfig = field(default_factory=CommunicationConfig)
    
    # Exploration strategies
    exploration_strategy: str = "epsilon_greedy"  # epsilon_greedy, ucb, thompson, noisy_nets, parameter_noise
    exploration_params: Dict[str, float] = field(default_factory=dict)
    
    # Curriculum learning
    curriculum_learning: bool = False
    curriculum_stages: List[Dict[str, Any]] = field(default_factory=list)
    
    # Individual learning features
    meta_learning: bool = False
    transfer_learning: bool = False
    continual_learning: bool = False
    
    # Evaluation and monitoring
    individual_evaluation: bool = True
    evaluation_episodes: int = 10
    evaluation_frequency: int = 1000
    
    # Custom parameters for research
    custom_params: Dict[str, Any] = field(default_factory=dict)
    research_tags: List[str] = field(default_factory=list)


@dataclass
class EnvironmentConfig:
    """Configuration for the environment."""
    env_name: str
    env_type: str = "MultiGrid"
    size: int = 15
    n_agents: int = 2
    max_steps: int = 100
    render_mode: Optional[str] = None
    
    # Environment-specific parameters
    env_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    max_episodes: int = 1000
    max_steps_per_episode: int = 100
    eval_interval: int = 100
    save_interval: int = 500
    log_interval: int = 10
    
    # Early stopping
    early_stopping: bool = False
    patience: int = 100
    min_improvement: float = 0.01
    
    # Logging and monitoring
    use_wandb: bool = False
    wandb_project: str = "easymarl"
    log_level: str = "INFO"
    save_videos: bool = False
    
    # Performance monitoring
    performance_threshold: Optional[float] = None
    convergence_window: int = 100


@dataclass
class ExperimentConfig:
    """Complete configuration for an experiment."""
    experiment_name: str
    algorithm: str
    environment: EnvironmentConfig
    training: TrainingConfig
    agents: List[AgentConfig]
    
    # Metadata
    description: str = ""
    tags: List[str] = field(default_factory=list)
    author: str = ""
    created_at: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = True
    
    # Resource management
    device: str = "auto"
    num_workers: int = 1
    
    # Hyperparameter optimization
    hp_search_space: Optional[Dict[str, Any]] = None
    hp_optimization: Optional[Dict[str, Any]] = None


class AlgorithmTemplates:
    """Pre-defined parameter templates for MARL algorithms."""
    
    @staticmethod
    def get_algorithm_template(algorithm: str) -> Dict[str, Any]:
        """Get the parameter template for a specific algorithm."""
        templates = {
            'ippo': AlgorithmTemplates._ippo_template(),
            'mappo': AlgorithmTemplates._mappo_template(),
            'maddpg': AlgorithmTemplates._maddpg_template(),
            'maddpgcomm': AlgorithmTemplates._maddpgcomm_template(),
            'qmix': AlgorithmTemplates._qmix_template(),
            'vdn': AlgorithmTemplates._vdn_template(),
            'iql': AlgorithmTemplates._iql_template(),
            'coma': AlgorithmTemplates._coma_template(),
            'comacomm': AlgorithmTemplates._comacomm_template(),
            'maacc': AlgorithmTemplates._maacc_template(),
            'maven': AlgorithmTemplates._maven_template(),
            'dcg': AlgorithmTemplates._dcg_template(),
            'nfsp': AlgorithmTemplates._nfsp_template(),
            'qtran': AlgorithmTemplates._qtran_template(),
            'mfq': AlgorithmTemplates._mfq_template(),
            'hql': AlgorithmTemplates._hql_template(),
            'lql': AlgorithmTemplates._lql_template()
        }
        
        return templates.get(algorithm.lower(), AlgorithmTemplates._default_template())
    
    @staticmethod
    def _ippo_template() -> Dict[str, Any]:
        """Independent PPO template."""
        return {
            'algorithm_type': 'policy_gradient',
            'learning_paradigm': 'independent',
            'action_space': 'discrete',
            'default_params': {
                'learning_rate': 3e-4,
                'gamma': 0.99,
                'clip_ratio': 0.2,
                'entropy_coef': 0.01,
                'value_coef': 0.5,
                'gae_lambda': 0.95,
                'ppo_epochs': 4,
                'batch_size': 64,
                'centralized_training': False,
                'decentralized_execution': True
            },
            'network_config': {
                'hidden_dims': [64, 64],
                'activation': 'tanh',
                'shared_networks': False,
                'parameter_sharing': 'none'
            },
            'memory_config': {
                'memory_type': 'replay_buffer',
                'shared_memory': False,
                'memory_size': 2048
            },
            'exploration': {
                'exploration_strategy': 'entropy_bonus',
                'exploration_params': {'entropy_coef': 0.01}
            }
        }
    
    @staticmethod
    def _mappo_template() -> Dict[str, Any]:
        """Multi-Agent PPO template."""
        return {
            'algorithm_type': 'policy_gradient',
            'learning_paradigm': 'centralized_training_decentralized_execution',
            'action_space': 'discrete',
            'default_params': {
                'learning_rate': 3e-4,
                'gamma': 0.99,
                'clip_ratio': 0.2,
                'entropy_coef': 0.01,
                'value_coef': 0.5,
                'gae_lambda': 0.95,
                'ppo_epochs': 4,
                'batch_size': 64,
                'centralized_training': True,
                'decentralized_execution': True
            },
            'network_config': {
                'hidden_dims': [128, 128],
                'activation': 'tanh',
                'shared_networks': True,
                'parameter_sharing': 'full'
            },
            'memory_config': {
                'memory_type': 'replay_buffer',
                'shared_memory': True,
                'shared_memory_ratio': 0.8,
                'memory_size': 4096
            },
            'exploration': {
                'exploration_strategy': 'entropy_bonus',
                'exploration_params': {'entropy_coef': 0.01}
            }
        }
    
    @staticmethod
    def _maddpg_template() -> Dict[str, Any]:
        """Multi-Agent DDPG template."""
        return {
            'algorithm_type': 'actor_critic',
            'learning_paradigm': 'centralized_training_decentralized_execution',
            'action_space': 'continuous',
            'default_params': {
                'actor_lr': 1e-3,
                'critic_lr': 1e-3,
                'gamma': 0.99,
                'tau': 0.005,
                'exploration_noise': 0.1,
                'action_noise_std': 0.2,
                'noise_clip': 0.5,
                'batch_size': 128,
                'centralized_training': True,
                'decentralized_execution': True
            },
            'network_config': {
                'hidden_dims': [128, 128],
                'activation': 'relu',
                'shared_networks': False,
                'parameter_sharing': 'none',
                'soft_update': True,
                'tau': 0.005
            },
            'memory_config': {
                'memory_type': 'replay_buffer',
                'shared_memory': False,
                'memory_size': 100000,
                'use_prioritized': True,
                'alpha': 0.6,
                'beta': 0.4
            },
            'exploration': {
                'exploration_strategy': 'parameter_noise',
                'exploration_params': {'noise_std': 0.2}
            }
        }
    
    @staticmethod
    def _maddpgcomm_template() -> Dict[str, Any]:
        """MADDPG with Communication template."""
        template = AlgorithmTemplates._maddpg_template()
        template['communication_config'] = {
            'enabled': True,
            'communication_type': 'direct',
            'message_size': 32,
            'max_messages': 3,
            'learnable_communication': True,
            'differentiable': True
        }
        return template
    
    @staticmethod
    def _qmix_template() -> Dict[str, Any]:
        """QMIX template."""
        return {
            'algorithm_type': 'value_based',
            'learning_paradigm': 'centralized_training_decentralized_execution',
            'action_space': 'discrete',
            'default_params': {
                'learning_rate': 5e-4,
                'gamma': 0.99,
                'epsilon': 1.0,
                'epsilon_min': 0.05,
                'epsilon_decay': 0.9995,
                'batch_size': 32,
                'target_update_freq': 200,
                'centralized_training': True,
                'decentralized_execution': True,
                'double_dqn': True
            },
            'network_config': {
                'hidden_dims': [64, 64],
                'activation': 'relu',
                'shared_networks': True,
                'parameter_sharing': 'full',
                'use_dueling': False
            },
            'memory_config': {
                'memory_type': 'replay_buffer',
                'shared_memory': True,
                'memory_size': 50000,
                'min_memory_size': 1000
            },
            'exploration': {
                'exploration_strategy': 'epsilon_greedy',
                'exploration_params': {'epsilon_schedule': 'exponential'}
            }
        }
    
    @staticmethod
    def _vdn_template() -> Dict[str, Any]:
        """VDN template."""
        template = AlgorithmTemplates._qmix_template()
        template['default_params']['target_update_freq'] = 100
        template['network_config']['hidden_dims'] = [64, 32]
        return template
    
    @staticmethod
    def _iql_template() -> Dict[str, Any]:
        """Independent Q-Learning template."""
        return {
            'algorithm_type': 'value_based',
            'learning_paradigm': 'independent',
            'action_space': 'discrete',
            'default_params': {
                'learning_rate': 1e-3,
                'gamma': 0.99,
                'epsilon': 1.0,
                'epsilon_min': 0.01,
                'epsilon_decay': 0.995,
                'batch_size': 32,
                'target_update_freq': 100,
                'centralized_training': False,
                'decentralized_execution': True,
                'double_dqn': False
            },
            'network_config': {
                'hidden_dims': [64, 64],
                'activation': 'relu',
                'shared_networks': False,
                'parameter_sharing': 'none'
            },
            'memory_config': {
                'memory_type': 'replay_buffer',
                'shared_memory': False,
                'memory_size': 10000
            },
            'exploration': {
                'exploration_strategy': 'epsilon_greedy',
                'exploration_params': {'epsilon_schedule': 'exponential'}
            }
        }
    
    @staticmethod
    def _coma_template() -> Dict[str, Any]:
        """COMA template."""
        return {
            'algorithm_type': 'actor_critic',
            'learning_paradigm': 'centralized_training_decentralized_execution',
            'action_space': 'discrete',
            'default_params': {
                'actor_lr': 5e-4,
                'critic_lr': 5e-4,
                'gamma': 0.99,
                'entropy_coef': 0.01,
                'batch_size': 8,
                'centralized_training': True,
                'decentralized_execution': True
            },
            'network_config': {
                'hidden_dims': [64, 64],
                'activation': 'relu',
                'shared_networks': False,
                'parameter_sharing': 'none'
            },
            'memory_config': {
                'memory_type': 'replay_buffer',
                'shared_memory': True,
                'memory_size': 5000
            },
            'exploration': {
                'exploration_strategy': 'entropy_bonus',
                'exploration_params': {'entropy_coef': 0.01}
            }
        }
    
    @staticmethod
    def _comacomm_template() -> Dict[str, Any]:
        """COMA with Communication template."""
        template = AlgorithmTemplates._coma_template()
        template['communication_config'] = {
            'enabled': True,
            'communication_type': 'broadcast',
            'message_size': 16,
            'max_messages': 2,
            'learnable_communication': True
        }
        return template
    
    @staticmethod
    def _maacc_template() -> Dict[str, Any]:
        """Multi-Agent Actor-Critic template."""
        return {
            'algorithm_type': 'actor_critic',
            'learning_paradigm': 'centralized_training_decentralized_execution',
            'action_space': 'discrete',
            'default_params': {
                'actor_lr': 1e-3,
                'critic_lr': 1e-3,
                'gamma': 0.99,
                'entropy_coef': 0.01,
                'value_coef': 0.5,
                'batch_size': 64,
                'centralized_training': True,
                'decentralized_execution': True
            },
            'network_config': {
                'hidden_dims': [128, 64],
                'activation': 'relu',
                'shared_networks': True,
                'parameter_sharing': 'partial'
            },
            'memory_config': {
                'memory_type': 'replay_buffer',
                'shared_memory': True,
                'memory_size': 20000
            }
        }
    
    @staticmethod
    def _maven_template() -> Dict[str, Any]:
        """MAVEN template."""
        template = AlgorithmTemplates._qmix_template()
        template['default_params'].update({
            'noise_dim': 16,
            'hierarchy_levels': 2,
            'exploration_bonus': 0.1
        })
        return template
    
    @staticmethod
    def _dcg_template() -> Dict[str, Any]:
        """Deep Coordination Graphs template."""
        return {
            'algorithm_type': 'value_based',
            'learning_paradigm': 'centralized_training_decentralized_execution',
            'action_space': 'discrete',
            'default_params': {
                'learning_rate': 5e-4,
                'gamma': 0.99,
                'epsilon': 1.0,
                'epsilon_min': 0.05,
                'epsilon_decay': 0.9995,
                'batch_size': 32,
                'coordination_graph_layers': 2,
                'centralized_training': True,
                'decentralized_execution': True
            },
            'network_config': {
                'hidden_dims': [128, 64],
                'activation': 'relu',
                'use_attention': True,
                'attention_heads': 4
            }
        }
    
    @staticmethod
    def _nfsp_template() -> Dict[str, Any]:
        """Neural Fictitious Self-Play template."""
        return {
            'algorithm_type': 'game_theoretic',
            'learning_paradigm': 'independent',
            'action_space': 'discrete',
            'default_params': {
                'learning_rate': 1e-3,
                'gamma': 0.99,
                'anticipatory_eta': 0.1,
                'reservoir_size': 2000000,
                'batch_size': 128,
                'centralized_training': False,
                'decentralized_execution': True
            },
            'network_config': {
                'hidden_dims': [128, 128],
                'activation': 'relu'
            },
            'memory_config': {
                'memory_type': 'reservoir',
                'shared_memory': False,
                'memory_size': 2000000
            }
        }
    
    @staticmethod
    def _qtran_template() -> Dict[str, Any]:
        """QTRAN template."""
        template = AlgorithmTemplates._qmix_template()
        template['default_params'].update({
            'opt_loss_weight': 1.0,
            'nopt_min_loss_weight': 0.1
        })
        return template
    
    @staticmethod
    def _mfq_template() -> Dict[str, Any]:
        """Mean Field Q-Learning template."""
        return {
            'algorithm_type': 'value_based',
            'learning_paradigm': 'mean_field',
            'action_space': 'discrete',
            'default_params': {
                'learning_rate': 1e-3,
                'gamma': 0.99,
                'epsilon': 0.1,
                'batch_size': 32,
                'temperature': 0.1,
                'centralized_training': False,
                'decentralized_execution': True
            },
            'network_config': {
                'hidden_dims': [64, 64],
                'activation': 'relu'
            }
        }
    
    @staticmethod
    def _hql_template() -> Dict[str, Any]:
        """Hysteretic Q-Learning template."""
        return {
            'algorithm_type': 'value_based',
            'learning_paradigm': 'independent',
            'action_space': 'discrete',
            'default_params': {
                'learning_rate': 0.1,
                'gamma': 0.9,
                'epsilon': 0.1,
                'hysteretic_learning_rate': 0.01,
                'centralized_training': False,
                'decentralized_execution': True
            },
            'tabular': True
        }
    
    @staticmethod
    def _lql_template() -> Dict[str, Any]:
        """Lenient Q-Learning template."""
        return {
            'algorithm_type': 'value_based',
            'learning_paradigm': 'independent',
            'action_space': 'discrete',
            'default_params': {
                'learning_rate': 0.1,
                'gamma': 0.9,
                'epsilon': 0.1,
                'leniency_temperature': 0.9,
                'centralized_training': False,
                'decentralized_execution': True
            },
            'tabular': True
        }
    
    @staticmethod
    def _default_template() -> Dict[str, Any]:
        """Default template for unknown algorithms."""
        return {
            'algorithm_type': 'unknown',
            'learning_paradigm': 'independent',
            'action_space': 'discrete',
            'default_params': {
                'learning_rate': 1e-3,
                'gamma': 0.99,
                'batch_size': 32,
                'centralized_training': False,
                'decentralized_execution': True
            },
            'network_config': {
                'hidden_dims': [64, 64],
                'activation': 'relu'
            },
            'memory_config': {
                'memory_type': 'replay_buffer',
                'shared_memory': False,
                'memory_size': 10000
            }
        }


class ConfigManager:
    """Manager for experiment configurations and templates."""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.templates_dir = self.config_dir / "templates"
        self.experiments_dir = self.config_dir / "experiments"
        self.presets_dir = self.config_dir / "presets"
        
        # Create directories if they don't exist
        for dir_path in [self.templates_dir, self.experiments_dir, self.presets_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def create_default_config(self, algorithm: str, n_agents: int = 2) -> ExperimentConfig:
        """Create a default configuration for the given algorithm using templates."""
        # Get algorithm template
        template = AlgorithmTemplates.get_algorithm_template(algorithm)
        
        # Create agent configs based on template
        agents = []
        for i in range(n_agents):
            agent_config = AgentConfig(agent_id=i)
            
            # Apply template defaults
            default_params = template.get('default_params', {})
            for param, value in default_params.items():
                if hasattr(agent_config, param):
                    setattr(agent_config, param, value)
            
            # Configure memory based on template
            if 'memory_config' in template:
                memory_cfg = template['memory_config']
                agent_config.memory = MemoryConfig(**memory_cfg)
            
            # Configure networks based on template
            if 'network_config' in template:
                network_cfg = template['network_config']
                agent_config.policy_network = NetworkConfig(**network_cfg)
                
                # Create separate value/critic networks if needed
                if template.get('algorithm_type') in ['actor_critic', 'value_based']:
                    agent_config.value_network = NetworkConfig(**network_cfg)
                
                if template.get('algorithm_type') == 'actor_critic':
                    agent_config.critic_network = NetworkConfig(**network_cfg)
            
            # Configure communication if specified
            if 'communication_config' in template:
                comm_cfg = template['communication_config']
                agent_config.communication = CommunicationConfig(**comm_cfg)
            
            # Set exploration based on template
            if 'exploration' in template:
                exploration = template['exploration']
                agent_config.exploration_strategy = exploration.get('exploration_strategy', 'epsilon_greedy')
                agent_config.exploration_params = exploration.get('exploration_params', {})
            
            agents.append(agent_config)
        
        # Create environment config
        env_config = EnvironmentConfig(
            env_name="MultiGrid-Cluttered-Fixed-15x15",
            n_agents=n_agents
        )
        
        # Create training config
        training_config = TrainingConfig()
        
        # Create experiment config
        experiment_config = ExperimentConfig(
            experiment_name=f"{algorithm}_template_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
            algorithm=algorithm,
            environment=env_config,
            training=training_config,
            agents=agents,
            description=f"Template configuration for {algorithm} algorithm based on research best practices",
            tags=[algorithm, template.get('algorithm_type', 'unknown'), template.get('learning_paradigm', 'independent')]
        )
        
        return experiment_config
    
    def get_algorithm_parameters(self, algorithm: str) -> Dict[str, Any]:
        """Get all configurable parameters for an algorithm."""
        template = AlgorithmTemplates.get_algorithm_template(algorithm)
        
        # Compile all possible parameters
        parameters = {
            'algorithm_info': {
                'type': template.get('algorithm_type', 'unknown'),
                'paradigm': template.get('learning_paradigm', 'independent'),
                'action_space': template.get('action_space', 'discrete'),
                'tabular': template.get('tabular', False)
            },
            'learning_parameters': {
                'learning_rate': {'type': 'float', 'range': [1e-5, 1e-1], 'default': 1e-3, 'log_scale': True},
                'gamma': {'type': 'float', 'range': [0.9, 0.999], 'default': 0.99},
                'batch_size': {'type': 'int', 'range': [8, 512], 'default': 32, 'power_of_2': True},
                'epsilon': {'type': 'float', 'range': [0.01, 1.0], 'default': 0.1},
                'epsilon_min': {'type': 'float', 'range': [0.001, 0.1], 'default': 0.01},
                'epsilon_decay': {'type': 'float', 'range': [0.99, 0.999], 'default': 0.995},
            },
            'network_parameters': {
                'hidden_dims': {'type': 'list_int', 'default': [64, 64], 'options': [[32], [64], [128], [32, 32], [64, 64], [128, 128], [64, 32], [128, 64], [256, 128]]},
                'activation': {'type': 'categorical', 'options': ['relu', 'tanh', 'gelu', 'swish'], 'default': 'relu'},
                'dropout': {'type': 'float', 'range': [0.0, 0.5], 'default': 0.0},
                'batch_norm': {'type': 'boolean', 'default': False},
                'layer_norm': {'type': 'boolean', 'default': False},
                'shared_networks': {'type': 'boolean', 'default': False},
                'parameter_sharing': {'type': 'categorical', 'options': ['none', 'full', 'partial'], 'default': 'none'}
            },
            'memory_parameters': {
                'memory_type': {'type': 'categorical', 'options': ['replay_buffer', 'prioritized_replay', 'shared_memory'], 'default': 'replay_buffer'},
                'memory_size': {'type': 'int', 'range': [1000, 1000000], 'default': 10000, 'log_scale': True},
                'shared_memory': {'type': 'boolean', 'default': False},
                'shared_memory_ratio': {'type': 'float', 'range': [0.1, 1.0], 'default': 0.7},
                'use_prioritized': {'type': 'boolean', 'default': False},
                'alpha': {'type': 'float', 'range': [0.0, 1.0], 'default': 0.6},
                'beta': {'type': 'float', 'range': [0.0, 1.0], 'default': 0.4},
                'experience_sharing': {'type': 'boolean', 'default': False}
            },
            'exploration_parameters': {
                'exploration_strategy': {'type': 'categorical', 'options': ['epsilon_greedy', 'ucb', 'thompson', 'noisy_nets', 'parameter_noise'], 'default': 'epsilon_greedy'},
                'exploration_noise': {'type': 'float', 'range': [0.01, 0.5], 'default': 0.1},
                'action_noise_std': {'type': 'float', 'range': [0.01, 1.0], 'default': 0.2}
            },
            'coordination_parameters': {
                'centralized_training': {'type': 'boolean', 'default': template.get('default_params', {}).get('centralized_training', False)},
                'decentralized_execution': {'type': 'boolean', 'default': template.get('default_params', {}).get('decentralized_execution', True)},
                'observation_sharing': {'type': 'boolean', 'default': False},
                'action_sharing': {'type': 'boolean', 'default': False},
                'reward_sharing': {'type': 'boolean', 'default': False}
            },
            'communication_parameters': {
                'enabled': {'type': 'boolean', 'default': False},
                'communication_type': {'type': 'categorical', 'options': ['direct', 'broadcast', 'selective', 'hierarchical'], 'default': 'direct'},
                'message_size': {'type': 'int', 'range': [8, 128], 'default': 32},
                'max_messages': {'type': 'int', 'range': [1, 10], 'default': 3},
                'learnable_communication': {'type': 'boolean', 'default': True},
                'differentiable': {'type': 'boolean', 'default': True}
            }
        }
        
        # Add algorithm-specific parameters
        if template.get('algorithm_type') == 'actor_critic':
            parameters['algorithm_specific'] = {
                'actor_lr': {'type': 'float', 'range': [1e-5, 1e-1], 'default': 1e-3, 'log_scale': True},
                'critic_lr': {'type': 'float', 'range': [1e-5, 1e-1], 'default': 1e-3, 'log_scale': True},
                'entropy_coef': {'type': 'float', 'range': [0.0, 0.1], 'default': 0.01},
                'value_coef': {'type': 'float', 'range': [0.1, 1.0], 'default': 0.5}
            }
        elif template.get('algorithm_type') == 'policy_gradient':
            parameters['algorithm_specific'] = {
                'clip_ratio': {'type': 'float', 'range': [0.1, 0.3], 'default': 0.2},
                'entropy_coef': {'type': 'float', 'range': [0.0, 0.1], 'default': 0.01},
                'value_coef': {'type': 'float', 'range': [0.1, 1.0], 'default': 0.5},
                'gae_lambda': {'type': 'float', 'range': [0.9, 0.99], 'default': 0.95},
                'ppo_epochs': {'type': 'int', 'range': [1, 10], 'default': 4}
            }
        elif template.get('algorithm_type') == 'value_based':
            parameters['algorithm_specific'] = {
                'target_update_freq': {'type': 'int', 'range': [10, 1000], 'default': 100},
                'double_dqn': {'type': 'boolean', 'default': False},
                'dueling_dqn': {'type': 'boolean', 'default': False},
                'n_step': {'type': 'int', 'range': [1, 10], 'default': 1}
            }
        
        # Apply template defaults
        template_params = template.get('default_params', {})
        for category in parameters:
            if isinstance(parameters[category], dict):
                for param_name, param_info in parameters[category].items():
                    if param_name in template_params:
                        param_info['default'] = template_params[param_name]
        
        return parameters
    
    def create_parameter_grid_from_template(self, algorithm: str, 
                                          param_ranges: Dict[str, List], 
                                          n_agents: int = 2) -> List[ExperimentConfig]:
        """Create a parameter grid using algorithm template as base."""
        base_config = self.create_default_config(algorithm, n_agents)
        return self.create_hyperparameter_grid(base_config, param_ranges)
    
    def save_config(self, config: ExperimentConfig, filepath: Optional[str] = None) -> str:
        """Save configuration to file."""
        if filepath is None:
            filename = f"{config.experiment_name}.yaml"
            filepath = self.experiments_dir / filename
        else:
            filepath = Path(filepath)
        
        # Convert to dictionary and save
        config_dict = asdict(config)
        
        with open(filepath, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        
        return str(filepath)
    
    def load_config(self, filepath: str) -> ExperimentConfig:
        """Load configuration from file."""
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Convert agent configs
        agents = [AgentConfig(**agent_dict) for agent_dict in config_dict['agents']]
        config_dict['agents'] = agents
        
        # Convert environment config
        config_dict['environment'] = EnvironmentConfig(**config_dict['environment'])
        
        # Convert training config
        config_dict['training'] = TrainingConfig(**config_dict['training'])
        
        return ExperimentConfig(**config_dict)
    
    def create_template(self, template_name: str, config: ExperimentConfig) -> str:
        """Save a configuration as a reusable template."""
        template_path = self.templates_dir / f"{template_name}.yaml"
        
        # Remove experiment-specific fields
        template_config = deepcopy(config)
        template_config.experiment_name = f"{template_name}_template"
        template_config.created_at = ""
        template_config.author = ""
        
        return self.save_config(template_config, template_path)
    
    def load_template(self, template_name: str) -> ExperimentConfig:
        """Load a configuration template."""
        template_path = self.templates_dir / f"{template_name}.yaml"
        return self.load_config(template_path)
    
    def list_templates(self) -> List[str]:
        """List available configuration templates."""
        return [f.stem for f in self.templates_dir.glob("*.yaml")]
    
    def list_experiments(self) -> List[str]:
        """List saved experiment configurations."""
        return [f.stem for f in self.experiments_dir.glob("*.yaml")]
    
    def create_hyperparameter_grid(self, base_config: ExperimentConfig, 
                                  param_grid: Dict[str, List[Any]]) -> List[ExperimentConfig]:
        """Create multiple configurations from a hyperparameter grid."""
        from itertools import product
        
        configs = []
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        for i, combination in enumerate(product(*param_values)):
            config = deepcopy(base_config)
            config.experiment_name = f"{base_config.experiment_name}_grid_{i:03d}"
            
            # Apply parameter combination
            for param_name, value in zip(param_names, combination):
                self._set_nested_param(config, param_name, value)
            
            configs.append(config)
        
        return configs
    
    def _set_nested_param(self, config: ExperimentConfig, param_path: str, value: Any):
        """Set a nested parameter in the configuration."""
        parts = param_path.split('.')
        
        if parts[0] == 'agents':
            # Handle agent-specific parameters
            if len(parts) >= 3:
                agent_idx = int(parts[1]) if parts[1].isdigit() else 0
                param_name = '.'.join(parts[2:])
                
                if agent_idx < len(config.agents):
                    self._set_object_param(config.agents[agent_idx], param_name, value)
        elif parts[0] in ['environment', 'training']:
            # Handle environment or training parameters
            obj = getattr(config, parts[0])
            param_name = '.'.join(parts[1:])
            self._set_object_param(obj, param_name, value)
        else:
            # Handle top-level parameters
            setattr(config, param_path, value)
    
    def _set_object_param(self, obj: Any, param_path: str, value: Any):
        """Set a parameter in an object, handling nested paths."""
        parts = param_path.split('.')
        
        if len(parts) == 1:
            setattr(obj, parts[0], value)
        else:
            # Handle nested objects (if needed in the future)
            sub_obj = getattr(obj, parts[0])
            self._set_object_param(sub_obj, '.'.join(parts[1:]), value)
    
    def create_algorithm_preset(self, algorithm: str, preset_name: str, 
                              params: Dict[str, Any]) -> str:
        """Create a preset parameter set for an algorithm."""
        preset_path = self.presets_dir / f"{algorithm}_{preset_name}.json"
        
        with open(preset_path, 'w') as f:
            json.dump(params, f, indent=2)
        
        return str(preset_path)
    
    def load_algorithm_preset(self, algorithm: str, preset_name: str) -> Dict[str, Any]:
        """Load a preset parameter set for an algorithm."""
        preset_path = self.presets_dir / f"{algorithm}_{preset_name}.json"
        
        with open(preset_path, 'r') as f:
            return json.load(f)
    
    def list_presets(self, algorithm: Optional[str] = None) -> List[str]:
        """List available algorithm presets."""
        if algorithm:
            pattern = f"{algorithm}_*.json"
        else:
            pattern = "*.json"
        
        presets = []
        for f in self.presets_dir.glob(pattern):
            preset_name = f.stem
            if algorithm:
                preset_name = preset_name.replace(f"{algorithm}_", "")
            presets.append(preset_name)
        
        return presets
    
    def validate_config(self, config: ExperimentConfig) -> List[str]:
        """Validate a configuration and return any issues found."""
        issues = []
        
        # Check required fields
        if not config.algorithm:
            issues.append("Algorithm not specified")
        
        if not config.agents:
            issues.append("No agents configured")
        
        if config.environment.n_agents != len(config.agents):
            issues.append(f"Environment expects {config.environment.n_agents} agents, "
                         f"but {len(config.agents)} agent configs provided")
        
        # Validate agent configs
        for i, agent in enumerate(config.agents):
            if agent.learning_rate <= 0:
                issues.append(f"Agent {i}: learning_rate must be positive")
            
            if not 0 <= agent.gamma <= 1:
                issues.append(f"Agent {i}: gamma must be between 0 and 1")
            
            if agent.epsilon < 0:
                issues.append(f"Agent {i}: epsilon cannot be negative")
        
        # Validate training config
        if config.training.max_episodes <= 0:
            issues.append("max_episodes must be positive")
        
        if config.training.max_steps_per_episode <= 0:
            issues.append("max_steps_per_episode must be positive")
        
        return issues
    
    def create_research_suite(self, base_config: ExperimentConfig, 
                            suite_name: str) -> Dict[str, List[ExperimentConfig]]:
        """Create a comprehensive research suite with multiple experiment variants."""
        suite = {}
        
        # Learning rate ablation
        lr_values = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
        suite['learning_rate_ablation'] = self.create_hyperparameter_grid(
            base_config, {'agents.0.learning_rate': lr_values}
        )
        
        # Epsilon exploration ablation (for value-based methods)
        if base_config.algorithm.lower() in ['dqn', 'iql', 'qmix', 'vdn']:
            epsilon_values = [0.05, 0.1, 0.2, 0.3, 0.5]
            suite['epsilon_ablation'] = self.create_hyperparameter_grid(
                base_config, {'agents.0.epsilon': epsilon_values}
            )
        
        # Network architecture ablation
        architectures = [
            [32, 32], [64, 64], [128, 128], 
            [64, 32], [128, 64], [256, 128]
        ]
        suite['architecture_ablation'] = self.create_hyperparameter_grid(
            base_config, {'agents.0.hidden_dims': architectures}
        )
        
        # Training duration experiments
        episode_counts = [500, 1000, 2000, 5000]
        suite['training_duration'] = self.create_hyperparameter_grid(
            base_config, {'training.max_episodes': episode_counts}
        )
        
        # Save the entire suite
        suite_dir = self.experiments_dir / suite_name
        suite_dir.mkdir(exist_ok=True)
        
        for experiment_type, configs in suite.items():
            type_dir = suite_dir / experiment_type
            type_dir.mkdir(exist_ok=True)
            
            for config in configs:
                config_path = type_dir / f"{config.experiment_name}.yaml"
                self.save_config(config, config_path)
        
        return suite


# Global instance for easy access
config_manager = ConfigManager()


def get_config_manager() -> ConfigManager:
    """Get the global configuration manager instance."""
    return config_manager
