"""
QTRAN (Q-Transformation) algorithm.

QTRAN relaxes the monotonicity constraint of QMIX by using additional regularization
to ensure Individual-Global-Max (IGM) principle through constraints rather than structure.
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


class QTRANAgent(MARLAgent):
    """
    QTRAN agent with individual Q-network.
    """
    
    def __init__(self, agent_id: int, obs_space: Dict, action_space: int, config: Dict):
        """
        Initialize the QTRAN agent.
        
        Args:
            agent_id: Unique identifier for this agent
            obs_space: Individual observation space
            action_space: Number of available actions
            config: Configuration dictionary
        """
        super().__init__(agent_id, obs_space, action_space, config)
        
        # QTRAN hyperparameters
        self.gamma = config.get('gamma', 0.99)
        self.epsilon_start = config.get('epsilon_start', 1.0)
        self.epsilon_end = config.get('epsilon_end', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.epsilon = self.epsilon_start
        
        # Individual Q-network
        self.q_network = QTRANQNetwork(obs_space, action_space, config).to(self.device)
        self.target_q_network = QTRANQNetwork(obs_space, action_space, config).to(self.device)
        
        # Copy weights to target network
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = Adam(self.q_network.parameters(), lr=config.get('learning_rate', 1e-3))
        
        print(f"Initialized QTRAN Agent {agent_id}")
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
    
    def get_target_q_values(self, observation: Dict) -> torch.Tensor:
        """Get target Q-values for all actions given observation."""
        obs_tensor = self._process_observation(observation)
        return self.target_q_network(obs_tensor)
    
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
        """Update target network weights."""
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
            'agent_id': self.agent_id,
            'config': self.config
        }
        torch.save(checkpoint, f"{path}_qtran_agent_{self.agent_id}.pth")
        print(f"Saved QTRAN Agent {self.agent_id} model to {path}_qtran_agent_{self.agent_id}.pth")
    
    def load_model(self, path: str):
        """Load the agent's model."""
        checkpoint = torch.load(f"{path}_qtran_agent_{self.agent_id}.pth", map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_q_network.load_state_dict(checkpoint['target_q_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        print(f"Loaded QTRAN Agent {self.agent_id} model from {path}_qtran_agent_{self.agent_id}.pth")


class QTRANQNetwork(nn.Module):
    """Individual Q-network for QTRAN agent."""
    
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


class QTRANMixingNetwork(nn.Module):
    """QTRAN mixing network that transforms individual Q-values to joint Q-value."""
    
    def __init__(self, n_agents: int, state_dim: int, config: Dict):
        super().__init__()
        
        self.n_agents = n_agents
        hidden_dim = config.get('hidden_dim', 128)
        
        # State embedding
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Individual Q-value transformation
        self.q_transform = nn.Sequential(
            nn.Linear(n_agents + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, individual_q_values: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Transform individual Q-values to joint Q-value.
        
        Args:
            individual_q_values: [batch_size, n_agents] individual Q-values
            state: [batch_size, state_dim] global state
            
        Returns:
            joint_q_value: [batch_size, 1] joint Q-value
        """
        batch_size = individual_q_values.shape[0]
        
        # Encode state
        state_features = self.state_encoder(state)  # [batch_size, hidden_dim]
        
        # Combine individual Q-values with state features
        combined = torch.cat([individual_q_values, state_features], dim=-1)  # [batch_size, n_agents + hidden_dim]
        
        # Transform to joint Q-value
        joint_q = self.q_transform(combined)  # [batch_size, 1]
        
        return joint_q


class QTRANCounterfactualNetwork(nn.Module):
    """Counterfactual value network for QTRAN regularization."""
    
    def __init__(self, n_agents: int, action_space: int, state_dim: int, config: Dict):
        super().__init__()
        
        self.n_agents = n_agents
        self.action_space = action_space
        hidden_dim = config.get('hidden_dim', 128)
        
        # State + action embedding
        self.input_dim = state_dim + n_agents * action_space
        
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Compute counterfactual value.
        
        Args:
            state: [batch_size, state_dim] global state
            actions: [batch_size, n_agents, action_space] one-hot actions
            
        Returns:
            counterfactual_value: [batch_size, 1]
        """
        batch_size = state.shape[0]
        
        # Flatten actions
        actions_flat = actions.view(batch_size, -1)  # [batch_size, n_agents * action_space]
        
        # Combine state and actions
        combined = torch.cat([state, actions_flat], dim=-1)  # [batch_size, state_dim + n_agents * action_space]
        
        return self.network(combined)


class QTRANEpisodeReplayBuffer:
    """Episode-based replay buffer for QTRAN."""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def add_episode(self, episode: Dict):
        """Add an episode to the buffer."""
        self.buffer.append(episode)
    
    def sample(self, batch_size: int) -> List[Dict]:
        """Sample episodes from the buffer."""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self):
        return len(self.buffer)


class QTRAN(MARLAlgorithm):
    """
    QTRAN (Q-Transformation) algorithm.
    
    Relaxes QMIX's monotonicity constraint through regularization terms
    that ensure the IGM principle through constraints rather than structure.
    """
    
    def __init__(self, env, config: Dict, device: torch.device):
        """
        Initialize QTRAN algorithm.
        
        Args:
            env: Multi-agent environment
            config: Configuration dictionary
            device: PyTorch device
        """
        super().__init__(env, config, device)
        
        # QTRAN specific parameters
        self.batch_size = config.get('batch_size', 32)
        self.target_update_freq = config.get('target_update_freq', 100)
        self.regularization_weight = config.get('regularization_weight', 0.1)
        
        # Get state dimension
        self.state_dim = self._get_state_dim()
        
        # Create mixing and counterfactual networks
        self.mixing_network = QTRANMixingNetwork(
            self.n_agents, self.state_dim, config
        ).to(device)
        
        self.target_mixing_network = QTRANMixingNetwork(
            self.n_agents, self.state_dim, config
        ).to(device)
        
        self.counterfactual_network = QTRANCounterfactualNetwork(
            self.n_agents, len(self.env.actions), self.state_dim, config
        ).to(device)
        
        # Copy weights to target networks
        self.target_mixing_network.load_state_dict(self.mixing_network.state_dict())
        
        # Optimizers
        self.mixing_optimizer = Adam(
            self.mixing_network.parameters(), 
            lr=config.get('learning_rate', 1e-3)
        )
        
        self.counterfactual_optimizer = Adam(
            self.counterfactual_network.parameters(),
            lr=config.get('learning_rate', 1e-3)
        )
        
        # Replay buffer
        self.replay_buffer = QTRANEpisodeReplayBuffer(config.get('memory_size', 10000))
        
        print(f"Initialized QTRAN")
        print(f"State dimension: {self.state_dim}")
        print(f"Mixing network parameters: {sum(p.numel() for p in self.mixing_network.parameters())}")
        print(f"Counterfactual network parameters: {sum(p.numel() for p in self.counterfactual_network.parameters())}")
    
    def _create_agents(self):
        """Create QTRAN agents."""
        obs_space = self._get_obs_space()
        action_space = len(self.env.actions)
        
        for i in range(self.n_agents):
            agent = QTRANAgent(i, obs_space, action_space, self.config)
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
    
    def _get_state_dim(self) -> int:
        """Get global state dimension."""
        obs_space = self._get_obs_space()
        obs_dim = 0
        for key, value in obs_space.items():
            if hasattr(value, 'shape'):
                obs_dim += np.prod(value.shape)
            elif isinstance(value, np.ndarray):
                obs_dim += np.prod(value.shape)
            else:
                obs_dim += 1
        
        return obs_dim * self.n_agents
    
    def _get_global_state(self, obs) -> np.ndarray:
        """Get global state from joint observations."""
        if isinstance(obs, dict):
            state_parts = []
            for i in range(self.n_agents):
                agent_obs = self._extract_agent_obs(obs, i)
                obs_flat = []
                for key, value in agent_obs.items():
                    if isinstance(value, np.ndarray):
                        obs_flat.append(value.flatten())
                    else:
                        obs_flat.append(np.array([value]))
                state_parts.append(np.concatenate(obs_flat))
            
            return np.concatenate(state_parts)
        else:
            return np.array(obs).flatten()
    
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
    
    def collect_episode(self, env) -> Dict:
        """Collect a complete episode for training."""
        obs = env.reset()
        done = False
        
        episode_data = {
            'observations': [[] for _ in range(self.n_agents)],
            'actions': [[] for _ in range(self.n_agents)],
            'rewards': [[] for _ in range(self.n_agents)],
            'states': [],
            'dones': []
        }
        
        while not done:
            # Get global state
            global_state = self._get_global_state(obs)
            episode_data['states'].append(global_state)
            
            # Get actions from all agents
            actions = []
            for i, agent in enumerate(self.agents):
                agent_obs = self._extract_agent_obs(obs, i)
                action, _ = agent.get_action(agent_obs, training=True)
                actions.append(action)
                
                # Store observations and actions
                episode_data['observations'][i].append(agent_obs)
                episode_data['actions'][i].append(action)
            
            # Take environment step
            next_obs, rewards, done, info = env.step(actions)
            
            # Store rewards and done
            if isinstance(rewards, list):
                for i in range(self.n_agents):
                    episode_data['rewards'][i].append(rewards[i])
            else:
                for i in range(self.n_agents):
                    episode_data['rewards'][i].append(rewards)
            
            episode_data['dones'].append(done)
            
            obs = next_obs
            self.total_steps += 1
        
        # Decay epsilon for all agents
        for agent in self.agents:
            agent.decay_epsilon()
        
        return episode_data
    
    def train_step(self, episode_data: Dict) -> Dict[str, float]:
        """Perform QTRAN training step."""
        # Add episode to replay buffer
        self.replay_buffer.add_episode(episode_data)
        
        if len(self.replay_buffer) < self.batch_size:
            return {'episode_length': len(episode_data['dones']),
                   'total_reward': sum([sum(episode_data['rewards'][i]) for i in range(self.n_agents)])}
        
        # Sample batch of episodes
        batch_episodes = self.replay_buffer.sample(self.batch_size)
        
        # Update networks
        losses = self._update_networks(batch_episodes)
        
        # Update target networks
        if self.total_steps % self.target_update_freq == 0:
            self._update_target_networks()
        
        losses.update({
            'episode_length': len(episode_data['dones']),
            'total_reward': sum([sum(episode_data['rewards'][i]) for i in range(self.n_agents)]),
            'epsilon': self.agents[0].epsilon,
            'total_steps': self.total_steps
        })
        
        return losses
    
    def _update_networks(self, batch_episodes: List[Dict]) -> Dict[str, float]:
        """Update QTRAN networks."""
        total_q_loss = 0.0
        total_regularization_loss = 0.0
        
        for episode in batch_episodes:
            episode_length = len(episode['states'])
            
            for t in range(episode_length - 1):
                # Current state and actions
                current_state = torch.tensor(episode['states'][t], dtype=torch.float32).unsqueeze(0).to(self.device)
                next_state = torch.tensor(episode['states'][t + 1], dtype=torch.float32).unsqueeze(0).to(self.device)
                
                # Get individual Q-values
                current_q_values = []
                next_q_values = []
                actions = []
                
                for i, agent in enumerate(self.agents):
                    # Current Q-values
                    agent_obs = episode['observations'][i][t]
                    current_q = agent.get_q_values(agent_obs)
                    current_q_values.append(current_q[0, episode['actions'][i][t]])
                    
                    # Next Q-values (target)
                    if t + 1 < len(episode['observations'][i]):
                        next_agent_obs = episode['observations'][i][t + 1]
                        next_q = agent.get_target_q_values(next_agent_obs)
                        next_q_values.append(torch.max(next_q))
                    else:
                        next_q_values.append(torch.tensor(0.0).to(self.device))
                    
                    # Actions (one-hot)
                    action_onehot = torch.zeros(len(self.env.actions)).to(self.device)
                    action_onehot[episode['actions'][i][t]] = 1.0
                    actions.append(action_onehot)
                
                current_q_values = torch.stack(current_q_values).unsqueeze(0)  # [1, n_agents]
                next_q_values = torch.stack(next_q_values).unsqueeze(0)  # [1, n_agents]
                actions_tensor = torch.stack(actions).unsqueeze(0)  # [1, n_agents, action_space]
                
                # Compute joint Q-values
                current_joint_q = self.mixing_network(current_q_values, current_state)
                next_joint_q = self.target_mixing_network(next_q_values, next_state)
                
                # Compute target
                rewards = torch.tensor([episode['rewards'][i][t] for i in range(self.n_agents)]).mean().unsqueeze(0).to(self.device)
                done = torch.tensor(episode['dones'][t], dtype=torch.float32).unsqueeze(0).to(self.device)
                target = rewards + self.gamma * next_joint_q * (1 - done)
                
                # Q-learning loss
                q_loss = F.mse_loss(current_joint_q, target.detach())
                
                # QTRAN regularization
                counterfactual_value = self.counterfactual_network(current_state, actions_tensor)
                
                # IGM constraint regularization
                individual_sum = current_q_values.sum(dim=-1, keepdim=True)
                regularization_loss = F.mse_loss(counterfactual_value, individual_sum)
                
                # Total loss
                total_loss = q_loss + self.regularization_weight * regularization_loss
                
                # Update mixing network
                self.mixing_optimizer.zero_grad()
                total_loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.mixing_network.parameters(), 1.0)
                self.mixing_optimizer.step()
                
                # Update counterfactual network
                self.counterfactual_optimizer.zero_grad()
                regularization_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.counterfactual_network.parameters(), 1.0)
                self.counterfactual_optimizer.step()
                
                # Update individual Q-networks
                for i, agent in enumerate(self.agents):
                    agent_obs = episode['observations'][i][t]
                    agent_q_values = agent.get_q_values(agent_obs)
                    agent_action = episode['actions'][i][t]
                    agent_q_value = agent_q_values[0, agent_action]
                    
                    # Individual Q-learning loss
                    agent_target = rewards + self.gamma * next_q_values[0, i] * (1 - done)
                    agent_loss = F.mse_loss(agent_q_value, agent_target.detach())
                    
                    agent.optimizer.zero_grad()
                    agent_loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.q_network.parameters(), 1.0)
                    agent.optimizer.step()
                
                total_q_loss += q_loss.item()
                total_regularization_loss += regularization_loss.item()
        
        return {
            'q_loss': total_q_loss / len(batch_episodes),
            'regularization_loss': total_regularization_loss / len(batch_episodes)
        }
    
    def _update_target_networks(self):
        """Update target networks."""
        self.target_mixing_network.load_state_dict(self.mixing_network.state_dict())
        for agent in self.agents:
            agent.update_target_network()
        
        print(f"Updated target networks at step {self.total_steps}")
