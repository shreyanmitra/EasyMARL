"""
EasyMARL Research Interface

This module provides a comprehensive research-oriented interface for MARL algorithm
discovery, experimentation, and hyperparameter optimization. Designed specifically
for researchers who need fine-grained control over algorithm parameters and
extensive experimental capabilities.

Key Research Features:
1. Algorithm Discovery: Browse and compare available algorithms
2. Dynamic Configuration: Real-time parameter adjustment through GUI
3. Individual Agent Tuning: Per-agent hyperparameter control
4. Experiment Management: Structured experiment tracking and comparison
5. Hyperparameter Search: Automated parameter optimization
6. Performance Analysis: Advanced metrics and visualization tools
7. Reproducibility: Comprehensive logging and experiment versioning

Created: August 17, 2025
Authors: EasyMARL Development Team
License: MIT License

Copyright (c) 2025 EasyMARL. All rights reserved.
"""

import json
import os
import itertools
import copy
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
import numpy as np
import torch
import wandb

# Import framework components
from algorithms.taxonomy import taxonomy, AlgorithmCategory, ModelFreeCategory
from algorithms import create_marl_algorithm, list_available_algorithms


@dataclass
class AgentConfig:
    """Configuration for an individual agent with research-friendly parameters."""
    
    # Core Learning Parameters
    learning_rate: float = 3e-4
    gamma: float = 0.99
    epsilon: float = 0.1
    
    # Network Architecture
    hidden_dims: List[int] = None
    activation: str = "relu"
    layer_norm: bool = False
    dropout: float = 0.0
    
    # Algorithm-Specific Parameters
    clip_ratio: float = 0.2  # PPO
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    gae_lambda: float = 0.95
    
    # Buffer/Memory Parameters
    buffer_size: int = 100000
    batch_size: int = 64
    sequence_length: int = 20  # For RNNs
    
    # Exploration Parameters
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.01
    exploration_noise: float = 0.1
    
    # Update Parameters
    update_frequency: int = 4
    target_update_frequency: int = 100
    grad_clip_norm: float = 0.5
    
    # Communication Parameters (for comm algorithms)
    comm_dim: int = 16
    comm_channels: int = 2
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [128, 128]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentConfig':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class ExperimentConfig:
    """Comprehensive experiment configuration for research."""
    
    # Experiment Metadata
    name: str = "research_experiment"
    description: str = ""
    tags: List[str] = None
    
    # Algorithm and Environment
    algorithm: str = "ippo"
    environment: str = "MultiGrid-Cluttered-Fixed-15x15"
    n_agents: int = 2
    
    # Training Parameters
    total_episodes: int = 10000
    max_steps_per_episode: int = 100
    eval_interval: int = 500
    eval_episodes: int = 10
    
    # Logging and Saving
    log_interval: int = 100
    save_interval: int = 1000
    visualize_interval: int = 1000
    use_wandb: bool = True
    wandb_project: str = "easymarl_research"
    
    # Device and Performance
    device: str = "auto"  # auto, cpu, cuda
    seed: int = 42
    num_workers: int = 1
    
    # Agent Configurations
    agent_configs: List[AgentConfig] = None
    shared_parameters: bool = True  # Whether agents share parameters
    
    # Research-Specific Features
    hyperparameter_search: bool = False
    search_space: Dict[str, Any] = None
    search_budget: int = 50  # Number of trials for hyperparameter search
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.agent_configs is None:
            self.agent_configs = [AgentConfig() for _ in range(self.n_agents)]
        if self.search_space is None:
            self.search_space = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['agent_configs'] = [config.to_dict() for config in self.agent_configs]
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentConfig':
        """Create from dictionary."""
        if 'agent_configs' in data:
            data['agent_configs'] = [
                AgentConfig.from_dict(config) for config in data['agent_configs']
            ]
        return cls(**data)


class AlgorithmDiscovery:
    """Research-oriented algorithm discovery and comparison system."""
    
    def __init__(self):
        self.taxonomy = taxonomy
        self.algorithms_info = self._build_algorithm_database()
    
    def _build_algorithm_database(self) -> Dict[str, Dict]:
        """Build comprehensive algorithm database with research metadata."""
        return {
            # Value-Based Approximation Methods
            'qmix': {
                'category': 'Value-Based → Approximation',
                'type': 'Cooperative',
                'action_space': 'Discrete',
                'paper': 'QMIX: Monotonic Value Function Factorisation (2018)',
                'key_concepts': ['Value Decomposition', 'Monotonic Networks', 'CTDE'],
                'hyperparameters': ['learning_rate', 'gamma', 'epsilon', 'batch_size', 'buffer_size'],
                'strengths': ['Theoretical guarantees', 'Good cooperative performance'],
                'limitations': ['Monotonic constraint', 'Limited to cooperative tasks'],
                'research_applications': ['Multi-robot coordination', 'Resource allocation'],
                'complexity': 'Intermediate',
                'sample_efficiency': 'Medium',
                'scalability': 'Good'
            },
            'vdn': {
                'category': 'Value-Based → Approximation',
                'type': 'Cooperative',
                'action_space': 'Discrete',
                'paper': 'Value-Decomposition Networks (2017)',
                'key_concepts': ['Additive Value Decomposition', 'Independent Learning'],
                'hyperparameters': ['learning_rate', 'gamma', 'epsilon', 'batch_size'],
                'strengths': ['Simple', 'Stable', 'Easy to implement'],
                'limitations': ['Additive constraint', 'Limited expressiveness'],
                'research_applications': ['Baseline comparisons', 'Simple coordination'],
                'complexity': 'Beginner',
                'sample_efficiency': 'Medium',
                'scalability': 'Excellent'
            },
            'qtran': {
                'category': 'Value-Based → Approximation',
                'type': 'Cooperative',
                'action_space': 'Discrete',
                'paper': 'QTRAN: Learning to Factorize (2019)',
                'key_concepts': ['General Value Decomposition', 'Counterfactual Reasoning'],
                'hyperparameters': ['learning_rate', 'gamma', 'opt_loss_weight', 'nopt_min_loss_weight'],
                'strengths': ['General decomposition', 'No monotonic constraint'],
                'limitations': ['Complex training', 'Hyperparameter sensitive'],
                'research_applications': ['Complex coordination', 'Mixed-motive games'],
                'complexity': 'Advanced',
                'sample_efficiency': 'Low',
                'scalability': 'Medium'
            },
            
            # Policy-Based Discrete Action Methods
            'ippo': {
                'category': 'Policy-Based → Discrete Action',
                'type': 'General Purpose',
                'action_space': 'Discrete',
                'paper': 'Independent PPO for Multi-Agent RL',
                'key_concepts': ['Independent Learning', 'Policy Gradients', 'Trust Regions'],
                'hyperparameters': ['learning_rate', 'clip_ratio', 'entropy_coef', 'gae_lambda'],
                'strengths': ['Stable', 'Easy to tune', 'Good baseline'],
                'limitations': ['No coordination', 'Non-stationarity'],
                'research_applications': ['Baseline', 'Independent agents', 'Comparison'],
                'complexity': 'Beginner',
                'sample_efficiency': 'High',
                'scalability': 'Excellent'
            },
            'mappo': {
                'category': 'Policy-Based → Discrete Action',
                'type': 'Cooperative',
                'action_space': 'Discrete',
                'paper': 'Multi-Agent PPO (2021)',
                'key_concepts': ['Centralized Training', 'Shared Critic', 'Parameter Sharing'],
                'hyperparameters': ['learning_rate', 'clip_ratio', 'entropy_coef', 'value_coef'],
                'strengths': ['State-of-the-art performance', 'Sample efficient'],
                'limitations': ['Requires global state', 'Complex implementation'],
                'research_applications': ['Cooperative tasks', 'Performance benchmarks'],
                'complexity': 'Advanced',
                'sample_efficiency': 'Very High',
                'scalability': 'Good'
            },
            
            # Policy-Based Continuous Action Methods
            'maddpg': {
                'category': 'Policy-Based → Continuous Action',
                'type': 'Mixed Cooperative/Competitive',
                'action_space': 'Continuous',
                'paper': 'Multi-Agent DDPG (2017)',
                'key_concepts': ['Actor-Critic', 'Continuous Control', 'Centralized Critics'],
                'hyperparameters': ['actor_lr', 'critic_lr', 'tau', 'exploration_noise'],
                'strengths': ['Continuous actions', 'Mixed scenarios'],
                'limitations': ['Sample inefficient', 'Unstable training'],
                'research_applications': ['Robotics', 'Continuous control', 'Competition'],
                'complexity': 'Advanced',
                'sample_efficiency': 'Low',
                'scalability': 'Medium'
            },
            
            # Actor-Critic Methods
            'coma': {
                'category': 'Actor-Critic',
                'type': 'Cooperative',
                'action_space': 'Discrete',
                'paper': 'Counterfactual Multi-Agent Policy Gradients (2018)',
                'key_concepts': ['Counterfactual Reasoning', 'Credit Assignment', 'Centralized Critic'],
                'hyperparameters': ['learning_rate', 'gamma', 'lambda', 'critic_lr'],
                'strengths': ['Sophisticated credit assignment', 'Theoretical foundation'],
                'limitations': ['Complex implementation', 'Computational overhead'],
                'research_applications': ['Credit assignment research', 'Cooperative AI'],
                'complexity': 'Expert',
                'sample_efficiency': 'Medium',
                'scalability': 'Poor'
            },
            
            # Tabular Methods
            'nashq': {
                'category': 'Value-Based → Tabular',
                'type': 'Competitive',
                'action_space': 'Discrete',
                'paper': 'Nash-Q Learning (1994)',
                'key_concepts': ['Game Theory', 'Nash Equilibrium', 'Minimax'],
                'hyperparameters': ['learning_rate', 'gamma', 'epsilon'],
                'strengths': ['Theoretical guarantees', 'Game-theoretic foundation'],
                'limitations': ['Tabular only', 'Computational complexity'],
                'research_applications': ['Game theory research', 'Competitive scenarios'],
                'complexity': 'Intermediate',
                'sample_efficiency': 'High',
                'scalability': 'Poor'
            }
        }
    
    def get_algorithm_info(self, algorithm: str) -> Dict[str, Any]:
        """Get comprehensive information about an algorithm."""
        return self.algorithms_info.get(algorithm, {})
    
    def search_algorithms(self, 
                         category: Optional[str] = None,
                         type: Optional[str] = None,
                         action_space: Optional[str] = None,
                         complexity: Optional[str] = None) -> List[str]:
        """Search algorithms by criteria."""
        results = []
        
        for algo, info in self.algorithms_info.items():
            if category and category not in info.get('category', ''):
                continue
            if type and info.get('type') != type:
                continue
            if action_space and info.get('action_space') != action_space:
                continue
            if complexity and info.get('complexity') != complexity:
                continue
            
            results.append(algo)
        
        return results
    
    def get_research_recommendations(self, 
                                   research_focus: str,
                                   experience_level: str = "intermediate") -> Dict[str, List[str]]:
        """Get algorithm recommendations based on research focus."""
        recommendations = {
            'cooperative_ai': {
                'beginner': ['vdn', 'ippo'],
                'intermediate': ['qmix', 'mappo'],
                'advanced': ['qtran', 'coma'],
                'expert': ['maven', 'dcg']
            },
            'competitive_ai': {
                'beginner': ['nashq', 'minimaxq'],
                'intermediate': ['wolfphc', 'nfsp'],
                'advanced': ['hql', 'lql'],
                'expert': ['mfq']
            },
            'continuous_control': {
                'beginner': ['maddpg'],
                'intermediate': ['maddpgcomm'],
                'advanced': ['dcg'],
                'expert': ['comacomm']
            },
            'sample_efficiency': {
                'beginner': ['ippo', 'mappo'],
                'intermediate': ['qmix', 'maven'],
                'advanced': ['qtran', 'coma'],
                'expert': ['maacc']
            },
            'scalability': {
                'beginner': ['vdn', 'ippo'],
                'intermediate': ['mfq', 'mappo'],
                'advanced': ['hql', 'lql'],
                'expert': ['dcg']
            }
        }
        
        return {
            'recommended': recommendations.get(research_focus, {}).get(experience_level, []),
            'alternatives': self._get_alternative_algorithms(research_focus, experience_level)
        }
    
    def _get_alternative_algorithms(self, research_focus: str, experience_level: str) -> List[str]:
        """Get alternative algorithms for comparison."""
        all_algos = list(self.algorithms_info.keys())
        return all_algos[:3]  # Simplified for now
    
    def compare_algorithms(self, algorithms: List[str]) -> Dict[str, Dict]:
        """Compare multiple algorithms across key dimensions."""
        comparison = {}
        
        for algo in algorithms:
            if algo in self.algorithms_info:
                info = self.algorithms_info[algo]
                comparison[algo] = {
                    'category': info.get('category', 'Unknown'),
                    'complexity': info.get('complexity', 'Unknown'),
                    'sample_efficiency': info.get('sample_efficiency', 'Unknown'),
                    'scalability': info.get('scalability', 'Unknown'),
                    'strengths': info.get('strengths', []),
                    'limitations': info.get('limitations', []),
                    'key_hyperparameters': info.get('hyperparameters', [])
                }
        
        return comparison


class HyperparameterOptimizer:
    """Research-grade hyperparameter optimization for MARL algorithms."""
    
    def __init__(self, experiment_config: ExperimentConfig):
        self.config = experiment_config
        self.trials = []
        self.best_trial = None
        self.search_history = []
    
    def define_search_space(self, algorithm: str) -> Dict[str, Any]:
        """Define algorithm-specific search spaces."""
        base_space = {
            'learning_rate': ('log_uniform', 1e-5, 1e-2),
            'gamma': ('uniform', 0.9, 0.999),
            'batch_size': ('choice', [32, 64, 128, 256]),
            'hidden_dims': ('choice', [[64, 64], [128, 128], [256, 256], [128, 64]])
        }
        
        algorithm_spaces = {
            'ippo': {
                **base_space,
                'clip_ratio': ('uniform', 0.1, 0.3),
                'entropy_coef': ('log_uniform', 1e-4, 1e-1),
                'gae_lambda': ('uniform', 0.9, 0.99)
            },
            'qmix': {
                **base_space,
                'epsilon': ('uniform', 0.05, 0.2),
                'target_update_frequency': ('choice', [50, 100, 200, 500])
            },
            'maddpg': {
                **base_space,
                'actor_lr': ('log_uniform', 1e-5, 1e-2),
                'critic_lr': ('log_uniform', 1e-5, 1e-2),
                'tau': ('uniform', 0.001, 0.01),
                'exploration_noise': ('uniform', 0.1, 0.3)
            }
        }
        
        return algorithm_spaces.get(algorithm, base_space)
    
    def sample_hyperparameters(self, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Sample hyperparameters from the search space."""
        params = {}
        
        for param, space_def in search_space.items():
            space_type, *args = space_def
            
            if space_type == 'uniform':
                params[param] = np.random.uniform(args[0], args[1])
            elif space_type == 'log_uniform':
                params[param] = np.exp(np.random.uniform(np.log(args[0]), np.log(args[1])))
            elif space_type == 'choice':
                params[param] = np.random.choice(args[0])
            elif space_type == 'int_uniform':
                params[param] = np.random.randint(args[0], args[1] + 1)
        
        return params
    
    def run_optimization(self, 
                        objective_metric: str = 'eval_mean_reward',
                        direction: str = 'maximize') -> Dict[str, Any]:
        """Run hyperparameter optimization."""
        print(f"Starting hyperparameter optimization for {self.config.algorithm}")
        print(f"Budget: {self.config.search_budget} trials")
        print(f"Objective: {direction} {objective_metric}")
        
        search_space = self.define_search_space(self.config.algorithm)
        best_score = float('-inf') if direction == 'maximize' else float('inf')
        
        for trial in range(self.config.search_budget):
            print(f"\nTrial {trial + 1}/{self.config.search_budget}")
            
            # Sample hyperparameters
            trial_params = self.sample_hyperparameters(search_space)
            
            # Update configuration with sampled parameters
            trial_config = self._create_trial_config(trial_params)
            
            # Run training
            try:
                results = self._run_trial(trial_config, trial)
                
                # Evaluate performance
                score = results.get(objective_metric, 0)
                
                # Update best trial
                is_better = (direction == 'maximize' and score > best_score) or \
                           (direction == 'minimize' and score < best_score)
                
                if is_better:
                    best_score = score
                    self.best_trial = {
                        'trial': trial,
                        'params': trial_params,
                        'score': score,
                        'results': results
                    }
                
                # Record trial
                self.trials.append({
                    'trial': trial,
                    'params': trial_params,
                    'score': score,
                    'results': results
                })
                
                print(f"Trial {trial + 1} score: {score:.4f}")
                if is_better:
                    print(f"New best score: {score:.4f}")
                
            except Exception as e:
                print(f"Trial {trial + 1} failed: {e}")
                continue
        
        # Log optimization results
        if self.best_trial:
            print(f"\nOptimization completed!")
            print(f"Best score: {self.best_trial['score']:.4f}")
            print(f"Best parameters: {self.best_trial['params']}")
        
        return self.best_trial
    
    def _create_trial_config(self, trial_params: Dict[str, Any]) -> ExperimentConfig:
        """Create experiment configuration for a trial."""
        trial_config = copy.deepcopy(self.config)
        
        # Update agent configurations with trial parameters
        for agent_config in trial_config.agent_configs:
            for param, value in trial_params.items():
                if hasattr(agent_config, param):
                    setattr(agent_config, param, value)
        
        # Update trial-specific settings
        trial_config.name = f"{self.config.name}_trial"
        trial_config.use_wandb = False  # Disable WandB for trials
        
        return trial_config
    
    def _run_trial(self, trial_config: ExperimentConfig, trial_num: int) -> Dict[str, Any]:
        """Run a single optimization trial."""
        # This would integrate with the main training loop
        # For now, return mock results
        return {
            'eval_mean_reward': np.random.normal(100, 20),
            'eval_std_reward': np.random.uniform(5, 15),
            'training_time': np.random.uniform(300, 600)
        }


class ResearchInterface:
    """Main research interface for the EasyMARL framework."""
    
    def __init__(self):
        self.discovery = AlgorithmDiscovery()
        self.current_config = ExperimentConfig()
        self.experiments = []
        self.current_experiment = None
    
    def create_experiment(self, 
                         name: str,
                         description: str = "",
                         algorithm: str = "ippo",
                         environment: str = "MultiGrid-Cluttered-Fixed-15x15") -> ExperimentConfig:
        """Create a new research experiment."""
        config = ExperimentConfig(
            name=name,
            description=description,
            algorithm=algorithm,
            environment=environment
        )
        
        self.current_config = config
        return config
    
    def get_algorithm_browser(self) -> Dict[str, Any]:
        """Get algorithm browser data for GUI."""
        algorithms = list_available_algorithms()
        categorized = {}
        
        for algo in algorithms:
            info = self.discovery.get_algorithm_info(algo)
            category = info.get('category', 'Other')
            
            if category not in categorized:
                categorized[category] = []
            
            categorized[category].append({
                'name': algo,
                'info': info
            })
        
        return categorized
    
    def get_parameter_schema(self, algorithm: str) -> Dict[str, Any]:
        """Get parameter schema for GUI generation."""
        schema = {
            'global_params': {
                'learning_rate': {
                    'type': 'float',
                    'default': 3e-4,
                    'min': 1e-6,
                    'max': 1e-1,
                    'step': 1e-5,
                    'description': 'Learning rate for neural network optimization'
                },
                'gamma': {
                    'type': 'float',
                    'default': 0.99,
                    'min': 0.0,
                    'max': 1.0,
                    'step': 0.01,
                    'description': 'Discount factor for future rewards'
                },
                'batch_size': {
                    'type': 'int',
                    'default': 64,
                    'min': 1,
                    'max': 512,
                    'step': 1,
                    'description': 'Batch size for training'
                }
            }
        }
        
        # Add algorithm-specific parameters
        if algorithm == 'ippo':
            schema['algorithm_params'] = {
                'clip_ratio': {
                    'type': 'float',
                    'default': 0.2,
                    'min': 0.1,
                    'max': 0.5,
                    'step': 0.01,
                    'description': 'PPO clipping ratio'
                },
                'entropy_coef': {
                    'type': 'float',
                    'default': 0.01,
                    'min': 0.001,
                    'max': 0.1,
                    'step': 0.001,
                    'description': 'Entropy coefficient for exploration'
                }
            }
        elif algorithm == 'qmix':
            schema['algorithm_params'] = {
                'epsilon': {
                    'type': 'float',
                    'default': 0.1,
                    'min': 0.01,
                    'max': 0.5,
                    'step': 0.01,
                    'description': 'Epsilon for epsilon-greedy exploration'
                },
                'target_update_frequency': {
                    'type': 'int',
                    'default': 100,
                    'min': 10,
                    'max': 1000,
                    'step': 10,
                    'description': 'Frequency of target network updates'
                }
            }
        
        return schema
    
    def validate_configuration(self, config: ExperimentConfig) -> List[str]:
        """Validate experiment configuration and return warnings/errors."""
        warnings = []
        
        # Check if algorithm exists
        if config.algorithm not in list_available_algorithms():
            warnings.append(f"Algorithm '{config.algorithm}' not found")
        
        # Check parameter ranges
        for agent_config in config.agent_configs:
            if agent_config.learning_rate <= 0:
                warnings.append("Learning rate must be positive")
            if not 0 <= agent_config.gamma <= 1:
                warnings.append("Gamma must be between 0 and 1")
        
        # Check computational requirements
        if config.total_episodes > 50000:
            warnings.append("Large number of episodes may take significant time")
        
        return warnings
    
    def save_experiment_config(self, config: ExperimentConfig, filepath: str) -> None:
        """Save experiment configuration to file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
        
        print(f"Configuration saved to {filepath}")
    
    def load_experiment_config(self, filepath: str) -> ExperimentConfig:
        """Load experiment configuration from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        config = ExperimentConfig.from_dict(data)
        self.current_config = config
        return config
    
    def start_hyperparameter_search(self, config: ExperimentConfig) -> Dict[str, Any]:
        """Start hyperparameter optimization."""
        optimizer = HyperparameterOptimizer(config)
        return optimizer.run_optimization()
    
    def get_experiment_comparison(self, experiment_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple experiments."""
        # This would load and compare experiment results
        return {
            'experiments': experiment_ids,
            'metrics_comparison': {},
            'hyperparameter_comparison': {},
            'performance_summary': {}
        }


# Global research interface instance
research_interface = ResearchInterface()


def get_research_interface() -> ResearchInterface:
    """Get the global research interface."""
    return research_interface


if __name__ == "__main__":
    # Example usage
    interface = ResearchInterface()
    
    # Create experiment
    config = interface.create_experiment(
        name="cooperative_coordination_study",
        description="Comparing value decomposition methods for cooperative tasks",
        algorithm="qmix"
    )
    
    # Get algorithm information
    algo_info = interface.discovery.get_algorithm_info("qmix")
    print("QMIX Information:")
    for key, value in algo_info.items():
        print(f"  {key}: {value}")
    
    # Get parameter schema for GUI
    schema = interface.get_parameter_schema("qmix")
    print("\nParameter Schema:")
    print(json.dumps(schema, indent=2))
