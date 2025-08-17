"""
Lenient Q-Learning (LQL) algorithm.

LQL addresses coordination problems in multi-agent environments by being
more forgiving of potentially sub-optimal joint actions during exploration.
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


class LQLAgent(MARLAgent):
    """
    Lenient Q-Learning agent with lenient updates.
    """
    
    def __init__(self, agent_id: int, obs_space: Dict, action_space: int, config: Dict):
        """
        Initialize the LQL agent.
        
        Args:
            agent_id: Unique identifier for this agent
            obs_space: Individual observation space
            action_space: Number of available actions
            config: Configuration dictionary
        """
        super().__init__(agent_id, obs_space, action_space, config)
        
        # LQL hyperparameters
        self.gamma = config.get('gamma', 0.99)
        self.epsilon_start = config.get('epsilon_start', 1.0)
        self.epsilon_end = config.get('epsilon_end', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.epsilon = self.epsilon_start
        self.learning_rate = config.get('learning_rate', 0.1)
        
        # Lenient parameters
        self.lenient_rate = config.get('lenient_rate', 0.8)  # Probability of lenient update
        self.temperature = config.get('temperature', 1.0)   # Temperature for lenient selection
        self.min_temp = config.get('min_temp', 0.1)        # Minimum temperature
        self.temp_decay = config.get('temp_decay', 0.999)  # Temperature decay
        
        # Q-network
        self.q_network = LQLQNetwork(obs_space, action_space, config).to(self.device)
        
        # Track maximum Q-values seen for each state-action pair
        self.max_q_values = {}
        
        print(f"Initialized LQL Agent {agent_id}")
        print(f"Lenient rate: {self.lenient_rate}")
        print(f"Temperature: {self.temperature}")
        print(f"Q-network parameters: {sum(p.numel() for p in self.q_network.parameters())}")
    
    def get_action(self, observation: Dict, training: bool = True) -> Tuple[int, float]:
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            observation: Individual agent observation
            training: Whether in training mode
            
        Returns:
            Tuple of (action, q_value)
        """
        with torch.no_grad():
            obs_tensor = self._process_observation(observation)
            q_values = self.q_network(obs_tensor)
            
            if training and random.random() < self.epsilon:
                action = random.randint(0, self.action_space - 1)
            else:
                action = torch.argmax(q_values, dim=-1).item()
            
            q_value = q_values[0, action].item()
            
            return action, q_value
    
    def get_q_values(self, observation: Dict) -> torch.Tensor:
        """Get Q-values for all actions given observation."""
        obs_tensor = self._process_observation(observation)
        return self.q_network(obs_tensor)
    
    def update(self, batch_data: Dict) -> Dict[str, float]:
        """
        Update Q-values using lenient learning.
        
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
        
        total_loss = 0.0
        lenient_updates = 0
        regular_updates = 0
        
        for i in range(len(observations)):
            # Current Q-value
            current_q = self.get_q_values(observations[i])
            current_q_value = current_q[0, actions[i]]
            
            # Next Q-value
            with torch.no_grad():
                next_q = self.get_q_values(next_observations[i])
                max_next_q = torch.max(next_q)
                
                # TD target
                target = rewards[i] + self.gamma * max_next_q * (1 - dones[i])
            
            # Get state key for tracking maximum Q-values
            state_key = self._get_state_key(observations[i])
            state_action_key = f"{state_key}_{actions[i]}"
            
            # Track maximum Q-value seen for this state-action pair
            if state_action_key not in self.max_q_values:
                self.max_q_values[state_action_key] = target.item()
            else:
                self.max_q_values[state_action_key] = max(
                    self.max_q_values[state_action_key], target.item()
                )
            
            # Lenient update decision
            if random.random() < self.lenient_rate:
                # Lenient update: use maximum between current target and historical max
                lenient_target = max(target.item(), self.max_q_values[state_action_key])
                lenient_target = torch.tensor(lenient_target).to(self.device)
                
                # Apply temperature-based soft maximum
                if self.temperature > self.min_temp:
                    weight = torch.sigmoid((lenient_target - target) / self.temperature)
                    final_target = weight * lenient_target + (1 - weight) * target
                else:
                    final_target = lenient_target
                
                lenient_updates += 1
            else:
                # Regular Q-learning update
                final_target = target
                regular_updates += 1
            
            # Compute loss
            loss = F.mse_loss(current_q_value, final_target.detach())
            
            # Manual gradient computation and update
            loss.backward()
            total_loss += loss.item()
        
        # Apply gradients with learning rate
        with torch.no_grad():
            for param in self.q_network.parameters():
                if param.grad is not None:
                    param -= self.learning_rate * param.grad / len(observations)
                    param.grad.zero_()
        
        # Decay temperature
        self.temperature = max(self.min_temp, self.temperature * self.temp_decay)
        
        return {
            'q_loss': total_loss / len(observations),
            'lenient_updates': lenient_updates,
            'regular_updates': regular_updates,
            'temperature': self.temperature,
            'epsilon': self.epsilon
        }
    
    def _get_state_key(self, observation: Dict) -> str:
        """Create a hashable key for the state."""
        if isinstance(observation, dict):
            state_parts = []
            for key, value in sorted(observation.items()):
                if isinstance(value, np.ndarray):
                    state_parts.append(f"{key}:{value.flatten().tolist()}")
                else:
                    state_parts.append(f"{key}:{value}")
            return "|".join(state_parts)
        else:
            return str(observation)
    
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
    
    def decay_epsilon(self):
        """Decay epsilon for exploration."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def reset_memory(self):
        """Reset agent memory."""
        self.memory = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'next_observations': [],
            'dones': []
        }
    
    def save_model(self, path: str):
        """Save the agent's model."""
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'epsilon': self.epsilon,
            'temperature': self.temperature,
            'max_q_values': self.max_q_values,
            'agent_id': self.agent_id,
            'config': self.config
        }
        torch.save(checkpoint, f"{path}_lql_agent_{self.agent_id}.pth")
        print(f"Saved LQL Agent {self.agent_id} model to {path}_lql_agent_{self.agent_id}.pth")
    
    def load_model(self, path: str):
        """Load the agent's model."""
        checkpoint = torch.load(f"{path}_lql_agent_{self.agent_id}.pth", map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.temperature = checkpoint['temperature']
        self.max_q_values = checkpoint['max_q_values']
        print(f"Loaded LQL Agent {self.agent_id} model from {path}_lql_agent_{self.agent_id}.pth")


class LQLQNetwork(nn.Module):
    """Q-network for Lenient Q-Learning agent."""
    
    def __init__(self, obs_space: Dict, action_space: int, config: Dict):
        super().__init__()
        
        # Estimate observation dimension
        self.obs_dim = self._estimate_obs_dim(obs_space)
        
        hidden_dim = config.get('hidden_dim', 128)
        
        self.network = nn.Sequential(
            nn.Linear(self.obs_dim, hidden_dim),
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
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass through Q-network."""
        return self.network(obs)


class LQLReplayBuffer:
    """Simple replay buffer for LQL."""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def add(self, transition: Dict):
        """Add a transition to the buffer."""
        self.buffer.append(transition)
    
    def sample(self, batch_size: int) -> Dict:
        """Sample a batch from the buffer."""
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
        # Organize into batches
        batch_data = {
            'observations': [t['observation'] for t in batch],
            'actions': [t['action'] for t in batch],
            'rewards': [t['reward'] for t in batch],
            'next_observations': [t['next_observation'] for t in batch],
            'dones': [t['done'] for t in batch]
        }
        
        return batch_data
    
    def __len__(self):
        return len(self.buffer)


class LQL(MARLAlgorithm):
    """
    Lenient Q-Learning (LQL) algorithm.
    
    Uses lenient updates that are more forgiving of potentially sub-optimal
    joint actions during exploration phases.
    """
    
    def __init__(self, env, config: Dict, device: torch.device):
        """
        Initialize LQL algorithm.
        
        Args:
            env: Multi-agent environment
            config: Configuration dictionary
            device: PyTorch device
        """
        super().__init__(env, config, device)
        
        # LQL specific parameters
        self.batch_size = config.get('batch_size', 32)
        self.memory_size = config.get('memory_size', 10000)
        self.train_start = config.get('train_start', 1000)
        self.update_interval = config.get('update_interval', 1)
        
        # Create replay buffers for each agent
        self.replay_buffers = [
            LQLReplayBuffer(self.memory_size) for _ in range(self.n_agents)
        ]
        
        print(f"Initialized LQL")
        print(f"Batch size: {self.batch_size}")
        print(f"Memory size: {self.memory_size}")
    
    def _create_agents(self):
        """Create LQL agents."""
        obs_space = self._get_obs_space()
        action_space = len(self.env.actions)
        
        for i in range(self.n_agents):
            agent = LQLAgent(i, obs_space, action_space, self.config)
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
        """Collect transitions and add to replay buffers."""
        obs = env.reset()
        done = False
        step_count = 0
        total_reward = [0.0] * self.n_agents
        
        while not done:
            # Get actions from all agents
            actions = []
            for i, agent in enumerate(self.agents):
                agent_obs = self._extract_agent_obs(obs, i)
                action, _ = agent.get_action(agent_obs, training=True)
                actions.append(action)
            
            # Take environment step
            next_obs, rewards, done, info = env.step(actions)
            
            # Store transitions in replay buffers
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
                    'done': done
                }
                
                self.replay_buffers[i].add(transition)
            
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
    
    def train_step(self, rollout_data: Dict) -> Dict[str, float]:
        """Perform LQL training step."""
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
            
            if agent_losses:
                # Average metrics across agents
                avg_metrics = {}
                for key in agent_losses[0].keys():
                    avg_metrics[f'avg_{key}'] = np.mean([loss[key] for loss in agent_losses])
                
                avg_metrics.update({
                    'episode_length': rollout_data['episode_length'],
                    'total_reward': sum(rollout_data['total_reward']) / self.n_agents,
                    'total_steps': self.total_steps
                })
                
                return avg_metrics
        
        return {
            'episode_length': rollout_data['episode_length'],
            'total_reward': sum(rollout_data['total_reward']) / self.n_agents,
            'total_steps': self.total_steps
        }
