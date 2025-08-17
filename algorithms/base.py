"""
Base classes for Multi-Agent Reinforcement Learning algorithms.

This module provides abstract base classes and common utilities for implementing
various MARL algorithms in a modular and extensible way.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Categorical
    TORCH_AVAILABLE = True
except ImportError:
    print("Warning: PyTorch not available. Install with: pip install torch")
    TORCH_AVAILABLE = False
    
    # Mock torch classes for code completion
    class torch:
        class Tensor: pass
        class device: pass
        @staticmethod
        def tensor(*args, **kwargs): pass
        @staticmethod
        def zeros(*args, **kwargs): pass
        @staticmethod
        def ones(*args, **kwargs): pass


class MARLAgent(ABC):
    """
    Abstract base class for all multi-agent reinforcement learning agents.
    
    This class defines the interface that all MARL agents must implement,
    ensuring consistency across different algorithm implementations.
    """
    
    def __init__(self, agent_id: int, obs_space: Dict, action_space: int, config: Dict):
        """
        Initialize the MARL agent.
        
        Args:
            agent_id: Unique identifier for this agent
            obs_space: Observation space specification
            action_space: Number of available actions
            config: Configuration dictionary containing hyperparameters
        """
        self.agent_id = agent_id
        self.obs_space = obs_space
        self.action_space = action_space
        self.config = config
        
        if TORCH_AVAILABLE:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = "cpu"
        
        # Initialize memory for storing experiences
        self.memory = {}
        self.reset_memory()
    
    @abstractmethod
    def get_action(self, observation: Dict, training: bool = True) -> Tuple[int, float]:
        """
        Select an action given the current observation.
        
        Args:
            observation: Current observation from the environment
            training: Whether the agent is in training mode
            
        Returns:
            Tuple of (action, log_probability)
        """
        pass
    
    @abstractmethod
    def update(self, batch_data: Dict) -> Dict[str, float]:
        """
        Update the agent's policy based on collected experiences.
        
        Args:
            batch_data: Dictionary containing batch of experiences
            
        Returns:
            Dictionary containing loss values and metrics
        """
        pass
    
    @abstractmethod
    def reset_memory(self):
        """Reset the agent's memory/buffer."""
        pass
    
    @abstractmethod
    def save_model(self, path: str):
        """Save the agent's model to disk."""
        pass
    
    @abstractmethod
    def load_model(self, path: str):
        """Load the agent's model from disk."""
        pass


class MARLAlgorithm(ABC):
    """
    Abstract base class for multi-agent reinforcement learning algorithms.
    
    This class manages multiple agents and coordinates their training.
    """
    
    def __init__(self, env, config: Dict, device):
        """
        Initialize the MARL algorithm.
        
        Args:
            env: Multi-agent environment
            config: Configuration dictionary
            device: PyTorch device (CPU/CUDA) or string
        """
        self.env = env
        self.config = config
        self.device = device
        self.n_agents = env.n_agents
        
        # Initialize agents
        self.agents = []
        self._create_agents()
        
        # Training statistics
        self.episode_count = 0
        self.total_steps = 0
        
    @abstractmethod
    def _create_agents(self):
        """Create and initialize all agents."""
        pass
    
    @abstractmethod
    def collect_rollout(self, env) -> Dict:
        """
        Collect a rollout of experiences from the environment.
        
        Args:
            env: The environment to collect from
            
        Returns:
            Dictionary containing the collected experiences
        """
        pass
    
    @abstractmethod
    def train_step(self, rollout_data: Dict) -> Dict[str, float]:
        """
        Perform one training step using the collected rollout data.
        
        Args:
            rollout_data: Data collected from environment rollouts
            
        Returns:
            Dictionary containing training metrics
        """
        pass
    
    def train(self, total_episodes: int) -> None:
        """
        Main training loop.
        
        Args:
            total_episodes: Total number of episodes to train for
        """
        for episode in range(total_episodes):
            # Collect rollout
            rollout_data = self.collect_rollout(self.env)
            
            # Train agents
            metrics = self.train_step(rollout_data)
            
            # Log progress
            if episode % self.config.get('log_interval', 100) == 0:
                self._log_metrics(episode, metrics)
            
            # Save models
            if episode % self.config.get('save_interval', 1000) == 0:
                self.save_models(f"episode_{episode}")
            
            self.episode_count += 1
    
    def _log_metrics(self, episode: int, metrics: Dict[str, float]):
        """Log training metrics."""
        print(f"Episode {episode}: {metrics}")
    
    def save_models(self, suffix: str = ""):
        """Save all agent models."""
        for i, agent in enumerate(self.agents):
            agent.save_model(f"agent_{i}_{suffix}")
    
    def load_models(self, suffix: str = ""):
        """Load all agent models."""
        for i, agent in enumerate(self.agents):
            agent.load_model(f"agent_{i}_{suffix}")


class ReplayBuffer:
    """
    Experience replay buffer for off-policy algorithms.
    """
    
    def __init__(self, capacity: int, obs_shape: Tuple, action_dim: int, n_agents: int):
        """
        Initialize the replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
            obs_shape: Shape of observations
            action_dim: Dimension of action space
            n_agents: Number of agents
        """
        self.capacity = capacity
        self.size = 0
        self.ptr = 0
        
        # Initialize buffers
        self.observations = np.zeros((capacity, n_agents) + obs_shape, dtype=np.float32)
        self.actions = np.zeros((capacity, n_agents, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, n_agents), dtype=np.float32)
        self.next_observations = np.zeros((capacity, n_agents) + obs_shape, dtype=np.float32)
        self.dones = np.zeros((capacity, n_agents), dtype=np.float32)
    
    def add(self, obs: np.ndarray, actions: np.ndarray, rewards: np.ndarray, 
            next_obs: np.ndarray, dones: np.ndarray):
        """Add a transition to the buffer."""
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = actions
        self.rewards[self.ptr] = rewards
        self.next_observations[self.ptr] = next_obs
        self.dones[self.ptr] = dones
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Dict[str, Any]:
        """Sample a batch of transitions."""
        indices = np.random.choice(self.size, batch_size, replace=False)
        
        batch = {
            'observations': self.observations[indices],
            'actions': self.actions[indices],
            'rewards': self.rewards[indices],
            'next_observations': self.next_observations[indices],
            'dones': self.dones[indices]
        }
        
        if TORCH_AVAILABLE:
            # Convert to tensors if PyTorch is available
            for key in batch:
                batch[key] = torch.FloatTensor(batch[key])
        
        return batch
    
    def __len__(self) -> int:
        return self.size


def compute_gae(rewards, values, next_values, dones, gamma: float = 0.99, lambda_: float = 0.95):
    """
    Compute Generalized Advantage Estimation (GAE).
    
    Args:
        rewards: Tensor of rewards [T, N]
        values: Tensor of value estimates [T, N]
        next_values: Tensor of next value estimates [T, N]
        dones: Tensor of done flags [T, N]
        gamma: Discount factor
        lambda_: GAE lambda parameter
        
    Returns:
        Tensor of GAE advantages [T, N]
    """
    if not TORCH_AVAILABLE:
        return np.zeros_like(rewards)
    
    advantages = torch.zeros_like(rewards)
    last_advantage = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = next_values[t]
        else:
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        advantages[t] = delta + gamma * lambda_ * (1 - dones[t]) * last_advantage
        last_advantage = advantages[t]
    
    return advantages


def compute_returns(rewards, values, dones, gamma: float = 0.99):
    """
    Compute discounted returns.
    
    Args:
        rewards: Tensor of rewards [T, N]
        values: Tensor of value estimates [T, N] 
        dones: Tensor of done flags [T, N]
        gamma: Discount factor
        
    Returns:
        Tensor of returns [T, N]
    """
    if not TORCH_AVAILABLE:
        return np.zeros_like(rewards)
    
    returns = torch.zeros_like(rewards)
    next_return = values[-1]  # Bootstrap from last value
    
    for t in reversed(range(len(rewards))):
        returns[t] = rewards[t] + gamma * next_return * (1 - dones[t])
        next_return = returns[t]
    
    return returns
