"""
Mean Field Q-Learning (MFQ) algorithm.

MFQ handles large-scale multi-agent systems by modeling interactions
through mean field approximations of the population dynamics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from typing import Dict, List, Tuple, Any
from collections import deque
import random

from .base import MARLAgent, MARLAlgorithm


class MFQAgent(MARLAgent):
    """
    Mean Field Q-Learning agent that models population interactions.
    """
    
    def __init__(self, agent_id: int, obs_space: Dict, action_space: int, config: Dict):
        """
        Initialize the MFQ agent.
        
        Args:
            agent_id: Unique identifier for this agent
            obs_space: Individual observation space
            action_space: Number of available actions
            config: Configuration dictionary
        """
        super().__init__(agent_id, obs_space, action_space, config)
        
        # MFQ hyperparameters
        self.gamma = config.get('gamma', 0.99)
        self.epsilon_start = config.get('epsilon_start', 1.0)
        self.epsilon_end = config.get('epsilon_end', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.epsilon = self.epsilon_start
        
        # Mean field parameters
        self.mean_field_dim = config.get('mean_field_dim', action_space)  # Action distribution dimension
        self.alpha = config.get('alpha', 0.1)  # Mean field update rate
        
        # Q-network with mean field input
        self.q_network = MFQNetwork(obs_space, action_space, self.mean_field_dim, config).to(self.device)
        self.target_q_network = MFQNetwork(obs_space, action_space, self.mean_field_dim, config).to(self.device)
        
        # Copy weights to target network
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = Adam(self.q_network.parameters(), lr=config.get('learning_rate', 1e-3))
        
        # Mean field state (action distribution of population)
        self.mean_field_state = torch.ones(self.mean_field_dim).to(self.device) / self.mean_field_dim
        self.mean_field_history = deque(maxlen=config.get('mf_history_length', 100))
        
        print(f"Initialized MFQ Agent {agent_id}")
        print(f"Q-network parameters: {sum(p.numel() for p in self.q_network.parameters())}")
        print(f"Mean field dimension: {self.mean_field_dim}")
    
    def get_action(self, observation: Dict, mean_field: torch.Tensor = None, training: bool = True) -> Tuple[int, float]:
        """
        Select an action using mean field Q-learning.
        
        Args:
            observation: Agent observation
            mean_field: Current mean field state (population action distribution)
            training: Whether in training mode
            
        Returns:
            Tuple of (action, q_value)
        """
        obs_tensor = self._process_observation(observation)
        
        # Use provided mean field or internal state
        if mean_field is None:
            mean_field = self.mean_field_state
        
        with torch.no_grad():
            if training and random.random() < self.epsilon:
                action = random.randint(0, self.action_space - 1)
                q_value = 0.0
            else:
                # Get Q-values conditioned on mean field
                q_values = self.q_network(obs_tensor, mean_field.unsqueeze(0))[0]
                action = torch.argmax(q_values).item()
                q_value = q_values[action].item()
        
        return action, q_value
    
    def update_mean_field(self, population_actions: List[int]):
        """Update mean field state based on population actions."""
        # Create action distribution
        action_counts = torch.zeros(self.mean_field_dim).to(self.device)
        for action in population_actions:
            if action < self.mean_field_dim:
                action_counts[action] += 1
        
        # Normalize to get distribution
        if len(population_actions) > 0:
            new_distribution = action_counts / len(population_actions)
        else:
            new_distribution = torch.ones(self.mean_field_dim).to(self.device) / self.mean_field_dim
        
        # Update mean field with momentum
        self.mean_field_state = (1 - self.alpha) * self.mean_field_state + self.alpha * new_distribution
        
        # Store in history
        self.mean_field_history.append(self.mean_field_state.clone())
    
    def get_mean_field_state(self) -> torch.Tensor:
        """Get current mean field state."""
        return self.mean_field_state
    
    def predict_mean_field(self, actions: List[int]) -> torch.Tensor:
        """Predict future mean field based on current actions."""
        # Simple prediction: weighted average of current and new distribution
        action_counts = torch.zeros(self.mean_field_dim).to(self.device)
        for action in actions:
            if action < self.mean_field_dim:
                action_counts[action] += 1
        
        if len(actions) > 0:
            new_distribution = action_counts / len(actions)
            predicted_mf = 0.7 * self.mean_field_state + 0.3 * new_distribution
        else:
            predicted_mf = self.mean_field_state
        
        return predicted_mf
    
    def update(self, batch_data: Dict) -> Dict[str, float]:
        """
        Update Q-network using mean field Q-learning.
        
        Args:
            batch_data: Dictionary containing transitions
            
        Returns:
            Dictionary containing loss metrics
        """
        observations = batch_data['observations']
        actions = batch_data['actions']
        rewards = batch_data['rewards']
        next_observations = batch_data['next_observations']
        dones = batch_data['dones']
        mean_fields = batch_data.get('mean_fields', [])
        next_mean_fields = batch_data.get('next_mean_fields', [])
        
        total_loss = 0.0
        
        for i in range(len(observations)):
            # Update with mean field information
            loss = self._update_q_network(
                observations[i], actions[i], rewards[i], 
                next_observations[i], dones[i],
                mean_fields[i] if i < len(mean_fields) else None,
                next_mean_fields[i] if i < len(next_mean_fields) else None
            )
            total_loss += loss
        
        return {
            'q_loss': total_loss / len(observations),
            'epsilon': self.epsilon,
            'mean_field_entropy': self._compute_entropy(self.mean_field_state)
        }
    
    def _update_q_network(self, observation: Dict, action: int, reward: float,
                         next_observation: Dict, done: bool, 
                         mean_field: torch.Tensor = None, next_mean_field: torch.Tensor = None) -> float:
        """Update Q-network with mean field target."""
        obs_tensor = self._process_observation(observation)
        next_obs_tensor = self._process_observation(next_observation)
        
        # Use current mean field if not provided
        if mean_field is None:
            mean_field = self.mean_field_state
        if next_mean_field is None:
            next_mean_field = self.mean_field_state
        
        # Current Q-value
        current_q = self.q_network(obs_tensor, mean_field.unsqueeze(0))[0, action]
        
        # Target Q-value with mean field
        with torch.no_grad():
            if not done:
                next_q_values = self.target_q_network(next_obs_tensor, next_mean_field.unsqueeze(0))[0]
                max_next_q = torch.max(next_q_values)
                target = reward + self.gamma * max_next_q
            else:
                target = torch.tensor(reward).to(self.device)
        
        # Mean field Q-learning loss
        loss = F.mse_loss(current_q, target)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def _compute_entropy(self, distribution: torch.Tensor) -> float:
        """Compute entropy of mean field distribution."""
        # Add small epsilon to avoid log(0)
        epsilon = 1e-8
        log_probs = torch.log(distribution + epsilon)
        entropy = -torch.sum(distribution * log_probs).item()
        return entropy
    
    def _process_observation(self, observation: Dict) -> torch.Tensor:
        """Convert observation to tensor format."""
        if isinstance(observation, dict):
            obs_list = []
            for key, value in observation.items():
                if isinstance(value, np.ndarray):
                    obs_list.append(torch.tensor(value, dtype=torch.float32).flatten())
                else:
                    obs_list.append(torch.tensor([value], dtype=torch.float32))
            obs_tensor = torch.cat(obs_list).unsqueeze(0).to(self.device)
        else:
            obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        return obs_tensor
    
    def update_target_network(self):
        """Update target network."""
        self.target_q_network.load_state_dict(self.q_network.state_dict())
    
    def decay_epsilon(self):
        """Decay epsilon for exploration."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save_model(self, path: str):
        """Save the agent's model."""
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_q_network_state_dict': self.target_q_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'mean_field_state': self.mean_field_state,
            'mean_field_history': list(self.mean_field_history),
            'agent_id': self.agent_id,
            'config': self.config
        }
        torch.save(checkpoint, f"{path}_mfq_agent_{self.agent_id}.pth")
        print(f"Saved MFQ Agent {self.agent_id} model to {path}_mfq_agent_{self.agent_id}.pth")
    
    def load_model(self, path: str):
        """Load the agent's model."""
        checkpoint = torch.load(f"{path}_mfq_agent_{self.agent_id}.pth", map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_q_network.load_state_dict(checkpoint['target_q_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.mean_field_state = checkpoint['mean_field_state']
        self.mean_field_history = deque(checkpoint['mean_field_history'], 
                                      maxlen=self.config.get('mf_history_length', 100))
        
        print(f"Loaded MFQ Agent {self.agent_id} model from {path}_mfq_agent_{self.agent_id}.pth")


class MFQNetwork(nn.Module):
    """Q-network that takes mean field as additional input."""
    
    def __init__(self, obs_space: Dict, action_space: int, mean_field_dim: int, config: Dict):
        super().__init__()
        
        self.obs_dim = self._estimate_obs_dim(obs_space)
        self.action_space = action_space
        self.mean_field_dim = mean_field_dim
        
        hidden_dim = config.get('hidden_dim', 256)
        
        # Observation encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(self.obs_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Mean field encoder
        self.mf_encoder = nn.Sequential(
            nn.Linear(mean_field_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Combined Q-network
        self.q_network = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_space)
        )
        
        self._init_weights()
    
    def _estimate_obs_dim(self, obs_space: Dict) -> int:
        """Estimate observation dimension."""
        total_dim = 0
        for key, value in obs_space.items():
            if hasattr(value, 'shape'):
                total_dim += np.prod(value.shape)
            elif isinstance(value, np.ndarray):
                total_dim += np.prod(value.shape)
            else:
                total_dim += 1
        return total_dim
    
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, obs: torch.Tensor, mean_field: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through mean field Q-network.
        
        Args:
            obs: Observation tensor
            mean_field: Mean field state (action distribution)
            
        Returns:
            Q-values for all actions
        """
        # Encode observation
        obs_features = self.obs_encoder(obs)
        
        # Encode mean field
        mf_features = self.mf_encoder(mean_field)
        
        # Combine features
        combined_features = torch.cat([obs_features, mf_features], dim=-1)
        
        # Get Q-values
        q_values = self.q_network(combined_features)
        
        return q_values


class MFQReplayBuffer:
    """Replay buffer for MFQ with mean field information."""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def add(self, transition: Dict):
        """Add a transition to the buffer."""
        self.buffer.append(transition)
    
    def sample(self, batch_size: int) -> Dict:
        """Sample a batch from the buffer."""
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
        batch_data = {
            'observations': [t['observation'] for t in batch],
            'actions': [t['action'] for t in batch],
            'rewards': [t['reward'] for t in batch],
            'next_observations': [t['next_observation'] for t in batch],
            'dones': [t['done'] for t in batch],
            'mean_fields': [t.get('mean_field') for t in batch],
            'next_mean_fields': [t.get('next_mean_field') for t in batch]
        }
        
        return batch_data
    
    def __len__(self):
        return len(self.buffer)


class MFQ(MARLAlgorithm):
    """
    Mean Field Q-Learning (MFQ) algorithm.
    
    Handles large-scale multi-agent systems through mean field approximations
    of population dynamics.
    """
    
    def __init__(self, env, config: Dict, device: torch.device):
        """
        Initialize MFQ algorithm.
        
        Args:
            env: Multi-agent environment
            config: Configuration dictionary
            device: PyTorch device
        """
        super().__init__(env, config, device)
        
        # MFQ specific parameters
        self.batch_size = config.get('batch_size', 32)
        self.memory_size = config.get('memory_size', 10000)
        self.train_start = config.get('train_start', 1000)
        self.update_interval = config.get('update_interval', 1)
        self.target_update_freq = config.get('target_update_freq', 100)
        
        # Create replay buffers for each agent
        self.replay_buffers = [
            MFQReplayBuffer(self.memory_size) for _ in range(self.n_agents)
        ]
        
        # Global mean field state
        self.global_mean_field = torch.ones(len(self.env.actions)).to(self.device) / len(self.env.actions)
        
        print(f"Initialized MFQ")
        print(f"Number of agents: {self.n_agents}")
        print(f"Mean field dimension: {len(self.env.actions)}")
        print(f"Batch size: {self.batch_size}")
    
    def _create_agents(self):
        """Create MFQ agents."""
        obs_space = self._get_obs_space()
        action_space = len(self.env.actions)
        
        for i in range(self.n_agents):
            agent = MFQAgent(i, obs_space, action_space, self.config)
            self.agents.append(agent)
    
    def _get_obs_space(self) -> Dict:
        """Get observation space specification."""
        sample_obs = self.env.reset()
        obs_space = {}
        
        if isinstance(sample_obs, dict):
            for key, value in sample_obs.items():
                if key == 'image' and isinstance(value, list):
                    obs_space[key] = value[0] if len(value) > 0 else np.zeros((7, 7, 3))
                elif key == 'direction' and isinstance(value, list):
                    obs_space[key] = value[0] if len(value) > 0 else 0
                else:
                    obs_space[key] = value
        
        return obs_space
    
    def _extract_agent_obs(self, obs, agent_id: int) -> Dict:
        """Extract observation for specific agent."""
        if isinstance(obs, dict):
            agent_obs = {}
            for key, value in obs.items():
                if isinstance(value, list):
                    agent_obs[key] = value[agent_id] if agent_id < len(value) else value[0]
                else:
                    agent_obs[key] = value
        else:
            agent_obs = obs[agent_id] if isinstance(obs, list) else obs
        
        return agent_obs
    
    def collect_rollout(self, env) -> Dict:
        """Collect rollout with mean field updates."""
        obs = env.reset()
        done = False
        step_count = 0
        total_reward = [0.0] * self.n_agents
        
        while not done:
            # Get actions from all agents
            actions = []
            current_mean_field = self.global_mean_field.clone()
            
            for i, agent in enumerate(self.agents):
                agent_obs = self._extract_agent_obs(obs, i)
                action, _ = agent.get_action(agent_obs, current_mean_field, training=True)
                actions.append(action)
            
            # Update mean field based on current actions
            self._update_global_mean_field(actions)
            next_mean_field = self.global_mean_field.clone()
            
            # Take environment step
            next_obs, rewards, done, info = env.step(actions)
            
            # Store transitions with mean field information
            for i in range(self.n_agents):
                agent_obs = self._extract_agent_obs(obs, i)
                next_agent_obs = self._extract_agent_obs(next_obs, i)
                
                reward = rewards[i] if isinstance(rewards, list) else rewards
                total_reward[i] += reward
                
                transition = {
                    'observation': agent_obs,
                    'action': actions[i],
                    'reward': reward,
                    'next_observation': next_agent_obs,
                    'done': done,
                    'mean_field': current_mean_field,
                    'next_mean_field': next_mean_field
                }
                
                self.replay_buffers[i].add(transition)
                
                # Update agent's local mean field
                self.agents[i].update_mean_field(actions)
            
            obs = next_obs
            step_count += 1
            self.total_steps += 1
        
        # Decay epsilon for all agents
        for agent in self.agents:
            agent.decay_epsilon()
        
        return {
            'episode_length': step_count,
            'total_reward': total_reward
        }
    
    def _update_global_mean_field(self, actions: List[int]):
        """Update global mean field based on population actions."""
        action_counts = torch.zeros(len(self.env.actions)).to(self.device)
        for action in actions:
            if action < len(self.env.actions):
                action_counts[action] += 1
        
        # Normalize to get distribution
        if len(actions) > 0:
            new_distribution = action_counts / len(actions)
            # Update with momentum
            alpha = 0.1
            self.global_mean_field = (1 - alpha) * self.global_mean_field + alpha * new_distribution
    
    def train_step(self, rollout_data: Dict) -> Dict[str, float]:
        """Perform MFQ training step."""
        if self.total_steps < self.train_start:
            return {
                'episode_length': rollout_data['episode_length'],
                'total_reward': sum(rollout_data['total_reward']) / self.n_agents
            }
        
        # Train each agent if we have enough experience
        if self.total_steps % self.update_interval == 0:
            agent_losses = []
            
            for i, agent in enumerate(self.agents):
                if len(self.replay_buffers[i]) >= self.batch_size:
                    # Sample batch and update agent
                    batch_data = self.replay_buffers[i].sample(self.batch_size)
                    loss_info = agent.update(batch_data)
                    agent_losses.append(loss_info)
            
            # Update target networks
            if self.total_steps % self.target_update_freq == 0:
                for agent in self.agents:
                    agent.update_target_network()
                print(f"Updated target networks at step {self.total_steps}")
            
            if agent_losses:
                # Average metrics across agents
                avg_metrics = {}
                for key in agent_losses[0].keys():
                    avg_metrics[f'avg_{key}'] = np.mean([loss[key] for loss in agent_losses])
                
                avg_metrics.update({
                    'episode_length': rollout_data['episode_length'],
                    'total_reward': sum(rollout_data['total_reward']) / self.n_agents,
                    'total_steps': self.total_steps,
                    'global_mean_field_entropy': self._compute_entropy(self.global_mean_field)
                })
                
                return avg_metrics
        
        return {
            'episode_length': rollout_data['episode_length'],
            'total_reward': sum(rollout_data['total_reward']) / self.n_agents,
            'total_steps': self.total_steps
        }
    
    def _compute_entropy(self, distribution: torch.Tensor) -> float:
        """Compute entropy of distribution."""
        epsilon = 1e-8
        log_probs = torch.log(distribution + epsilon)
        entropy = -torch.sum(distribution * log_probs).item()
        return entropy
