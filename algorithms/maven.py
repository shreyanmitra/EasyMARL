"""
MAVEN (Multi-Agent Variational Exploration) algorithm.

MAVEN uses mutual information maximization to learn diverse behaviors
and improve exploration in multi-agent environments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from torch.optim import Adam
import numpy as np
from typing import Dict, List, Tuple, Any
from collections import deque
import random

from .base import MARLAgent, MARLAlgorithm


class MAVENAgent(MARLAgent):
    """
    MAVEN agent with latent variable for exploration.
    """
    
    def __init__(self, agent_id: int, obs_space: Dict, action_space: int, config: Dict):
        """
        Initialize the MAVEN agent.
        
        Args:
            agent_id: Unique identifier for this agent
            obs_space: Individual observation space
            action_space: Number of available actions
            config: Configuration dictionary
        """
        super().__init__(agent_id, obs_space, action_space, config)
        
        # MAVEN hyperparameters
        self.gamma = config.get('gamma', 0.99)
        self.epsilon_start = config.get('epsilon_start', 1.0)
        self.epsilon_end = config.get('epsilon_end', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.epsilon = self.epsilon_start
        
        # Latent space parameters
        self.latent_dim = config.get('latent_dim', 8)
        self.noise_scale = config.get('noise_scale', 0.3)
        
        # Networks
        self.q_network = MAVENQNetwork(obs_space, action_space, self.latent_dim, config).to(self.device)
        self.target_q_network = MAVENQNetwork(obs_space, action_space, self.latent_dim, config).to(self.device)
        
        # Copy weights to target network
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = Adam(self.q_network.parameters(), lr=config.get('learning_rate', 1e-3))
        
        # Current latent variable
        self.current_latent = None
        
        print(f"Initialized MAVEN Agent {agent_id}")
        print(f"Latent dimension: {self.latent_dim}")
        print(f"Q-network parameters: {sum(p.numel() for p in self.q_network.parameters())}")
    
    def set_latent(self, latent: torch.Tensor):
        """Set the current latent variable."""
        self.current_latent = latent.to(self.device)
    
    def get_action(self, observation: Dict, training: bool = True) -> Tuple[int, float]:
        """
        Select an action given the current observation and latent variable.
        
        Args:
            observation: Individual agent observation
            training: Whether in training mode
            
        Returns:
            Tuple of (action, q_value)
        """
        if self.current_latent is None:
            # Use default latent if none set
            self.current_latent = torch.zeros(1, self.latent_dim).to(self.device)
        
        with torch.no_grad():
            obs_tensor = self._process_observation(observation)
            q_values = self.q_network(obs_tensor, self.current_latent)
            
            if training and random.random() < self.epsilon:
                action = random.randint(0, self.action_space - 1)
            else:
                action = torch.argmax(q_values, dim=-1).item()
            
            q_value = q_values[0, action].item()
            
            return action, q_value
    
    def get_q_values(self, observation: Dict, latent: torch.Tensor) -> torch.Tensor:
        """Get Q-values for all actions given observation and latent."""
        obs_tensor = self._process_observation(observation)
        return self.q_network(obs_tensor, latent)
    
    def get_target_q_values(self, observation: Dict, latent: torch.Tensor) -> torch.Tensor:
        """Get target Q-values for all actions given observation and latent."""
        obs_tensor = self._process_observation(observation)
        return self.target_q_network(obs_tensor, latent)
    
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
        torch.save(checkpoint, f"{path}_maven_agent_{self.agent_id}.pth")
        print(f"Saved MAVEN Agent {self.agent_id} model to {path}_maven_agent_{self.agent_id}.pth")
    
    def load_model(self, path: str):
        """Load the agent's model."""
        checkpoint = torch.load(f"{path}_maven_agent_{self.agent_id}.pth", map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_q_network.load_state_dict(checkpoint['target_q_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        print(f"Loaded MAVEN Agent {self.agent_id} model from {path}_maven_agent_{self.agent_id}.pth")


class MAVENQNetwork(nn.Module):
    """Q-network that takes observation and latent variable as input."""
    
    def __init__(self, obs_space: Dict, action_space: int, latent_dim: int, config: Dict):
        super().__init__()
        
        # Estimate observation dimension
        self.obs_dim = self._estimate_obs_dim(obs_space)
        self.latent_dim = latent_dim
        
        hidden_dim = config.get('hidden_dim', 128)
        
        # Observation encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(self.obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Latent encoder
        self.latent_encoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Combined Q-value head
        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
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
    
    def forward(self, obs: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Q-network.
        
        Args:
            obs: [batch_size, obs_dim] observations
            latent: [batch_size, latent_dim] latent variables
            
        Returns:
            Q-values: [batch_size, action_space]
        """
        # Encode observation and latent
        obs_features = self.obs_encoder(obs)
        latent_features = self.latent_encoder(latent)
        
        # Combine features
        combined = torch.cat([obs_features, latent_features], dim=-1)
        
        # Get Q-values
        q_values = self.q_head(combined)
        
        return q_values


class MAVENMixingNetwork(nn.Module):
    """Mixing network that combines individual Q-values with latent variable."""
    
    def __init__(self, n_agents: int, state_dim: int, latent_dim: int, config: Dict):
        super().__init__()
        
        self.n_agents = n_agents
        self.latent_dim = latent_dim
        hidden_dim = config.get('hidden_dim', 128)
        
        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Latent encoder
        self.latent_encoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Mixing network
        self.mixer = nn.Sequential(
            nn.Linear(n_agents + hidden_dim + hidden_dim, hidden_dim),
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
    
    def forward(self, individual_q_values: torch.Tensor, state: torch.Tensor, 
               latent: torch.Tensor) -> torch.Tensor:
        """
        Mix individual Q-values with state and latent information.
        
        Args:
            individual_q_values: [batch_size, n_agents] individual Q-values
            state: [batch_size, state_dim] global state
            latent: [batch_size, latent_dim] latent variable
            
        Returns:
            joint_q_value: [batch_size, 1] joint Q-value
        """
        # Encode state and latent
        state_features = self.state_encoder(state)
        latent_features = self.latent_encoder(latent)
        
        # Combine all features
        combined = torch.cat([individual_q_values, state_features, latent_features], dim=-1)
        
        # Mix to get joint Q-value
        joint_q = self.mixer(combined)
        
        return joint_q


class MAVENVariationalNetwork(nn.Module):
    """Variational network for latent variable inference."""
    
    def __init__(self, state_dim: int, latent_dim: int, config: Dict):
        super().__init__()
        
        self.latent_dim = latent_dim
        hidden_dim = config.get('hidden_dim', 128)
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Mean and variance heads
        self.mean_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode state to latent distribution parameters.
        
        Args:
            state: [batch_size, state_dim] global state
            
        Returns:
            Tuple of (mean, logvar) for latent distribution
        """
        features = self.encoder(state)
        
        mean = self.mean_head(features)
        logvar = self.logvar_head(features)
        
        return mean, logvar
    
    def sample(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample from the latent distribution using reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std


class MAVENEpisodeReplayBuffer:
    """Episode-based replay buffer for MAVEN."""
    
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


class MAVEN(MARLAlgorithm):
    """
    MAVEN (Multi-Agent Variational Exploration) algorithm.
    
    Uses mutual information maximization to learn diverse behaviors
    and improve exploration in multi-agent environments.
    """
    
    def __init__(self, env, config: Dict, device: torch.device):
        """
        Initialize MAVEN algorithm.
        
        Args:
            env: Multi-agent environment
            config: Configuration dictionary
            device: PyTorch device
        """
        super().__init__(env, config, device)
        
        # MAVEN specific parameters
        self.batch_size = config.get('batch_size', 32)
        self.target_update_freq = config.get('target_update_freq', 100)
        self.latent_dim = config.get('latent_dim', 8)
        self.mi_loss_weight = config.get('mi_loss_weight', 0.001)
        self.kl_loss_weight = config.get('kl_loss_weight', 0.0001)
        
        # Get state dimension
        self.state_dim = self._get_state_dim()
        
        # Create networks
        self.mixing_network = MAVENMixingNetwork(
            self.n_agents, self.state_dim, self.latent_dim, config
        ).to(device)
        
        self.target_mixing_network = MAVENMixingNetwork(
            self.n_agents, self.state_dim, self.latent_dim, config
        ).to(device)
        
        self.variational_network = MAVENVariationalNetwork(
            self.state_dim, self.latent_dim, config
        ).to(device)
        
        # Copy weights to target network
        self.target_mixing_network.load_state_dict(self.mixing_network.state_dict())
        
        # Optimizers
        self.mixing_optimizer = Adam(
            self.mixing_network.parameters(), 
            lr=config.get('learning_rate', 1e-3)
        )
        
        self.variational_optimizer = Adam(
            self.variational_network.parameters(),
            lr=config.get('learning_rate', 1e-3)
        )
        
        # Replay buffer
        self.replay_buffer = MAVENEpisodeReplayBuffer(config.get('memory_size', 10000))
        
        print(f"Initialized MAVEN")
        print(f"Latent dimension: {self.latent_dim}")
        print(f"State dimension: {self.state_dim}")
        print(f"Mixing network parameters: {sum(p.numel() for p in self.mixing_network.parameters())}")
        print(f"Variational network parameters: {sum(p.numel() for p in self.variational_network.parameters())}")
    
    def _create_agents(self):
        """Create MAVEN agents."""
        obs_space = self._get_obs_space()
        action_space = len(self.env.actions)
        
        for i in range(self.n_agents):
            agent = MAVENAgent(i, obs_space, action_space, self.config)
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
    
    def _sample_latent(self, batch_size: int = 1) -> torch.Tensor:
        """Sample random latent variable."""
        return torch.randn(batch_size, self.latent_dim).to(self.device)
    
    def collect_episode(self, env) -> Dict:
        """Collect a complete episode with latent variable."""
        obs = env.reset()
        done = False
        
        # Sample latent variable for this episode
        episode_latent = self._sample_latent(1)
        
        # Set latent for all agents
        for agent in self.agents:
            agent.set_latent(episode_latent)
        
        episode_data = {
            'observations': [[] for _ in range(self.n_agents)],
            'actions': [[] for _ in range(self.n_agents)],
            'rewards': [[] for _ in range(self.n_agents)],
            'states': [],
            'dones': [],
            'latent': episode_latent
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
        """Perform MAVEN training step."""
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
        """Update MAVEN networks."""
        total_q_loss = 0.0
        total_mi_loss = 0.0
        total_kl_loss = 0.0
        
        for episode in batch_episodes:
            episode_length = len(episode['states'])
            episode_latent = episode['latent']
            
            for t in range(episode_length - 1):
                # Current state
                current_state = torch.tensor(episode['states'][t], dtype=torch.float32).unsqueeze(0).to(self.device)
                next_state = torch.tensor(episode['states'][t + 1], dtype=torch.float32).unsqueeze(0).to(self.device)
                
                # Get individual Q-values with latent
                current_q_values = []
                next_q_values = []
                
                for i, agent in enumerate(self.agents):
                    # Current Q-values
                    agent_obs = episode['observations'][i][t]
                    current_q = agent.get_q_values(agent_obs, episode_latent)
                    current_q_values.append(current_q[0, episode['actions'][i][t]])
                    
                    # Next Q-values (target)
                    if t + 1 < len(episode['observations'][i]):
                        next_agent_obs = episode['observations'][i][t + 1]
                        next_q = agent.get_target_q_values(next_agent_obs, episode_latent)
                        next_q_values.append(torch.max(next_q))
                    else:
                        next_q_values.append(torch.tensor(0.0).to(self.device))
                
                current_q_values = torch.stack(current_q_values).unsqueeze(0)  # [1, n_agents]
                next_q_values = torch.stack(next_q_values).unsqueeze(0)  # [1, n_agents]
                
                # Compute joint Q-values
                current_joint_q = self.mixing_network(current_q_values, current_state, episode_latent)
                next_joint_q = self.target_mixing_network(next_q_values, next_state, episode_latent)
                
                # Compute target
                rewards = torch.tensor([episode['rewards'][i][t] for i in range(self.n_agents)]).mean().unsqueeze(0).to(self.device)
                done = torch.tensor(episode['dones'][t], dtype=torch.float32).unsqueeze(0).to(self.device)
                target = rewards + self.gamma * next_joint_q * (1 - done)
                
                # Q-learning loss
                q_loss = F.mse_loss(current_joint_q, target.detach())
                
                # Mutual information loss
                # Infer latent from state
                inferred_mean, inferred_logvar = self.variational_network(current_state)
                
                # MI loss (maximize mutual information between latent and state)
                inferred_std = torch.exp(0.5 * inferred_logvar)
                inferred_dist = Normal(inferred_mean, inferred_std)
                log_prob = inferred_dist.log_prob(episode_latent).sum(dim=-1, keepdim=True)
                mi_loss = -log_prob.mean()
                
                # KL regularization (prevent latent collapse)
                kl_loss = -0.5 * torch.sum(1 + inferred_logvar - inferred_mean.pow(2) - inferred_logvar.exp())
                
                # Total loss
                total_loss = q_loss + self.mi_loss_weight * mi_loss + self.kl_loss_weight * kl_loss
                
                # Update mixing network
                self.mixing_optimizer.zero_grad()
                total_loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.mixing_network.parameters(), 1.0)
                self.mixing_optimizer.step()
                
                # Update variational network
                var_loss = self.mi_loss_weight * mi_loss + self.kl_loss_weight * kl_loss
                self.variational_optimizer.zero_grad()
                var_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.variational_network.parameters(), 1.0)
                self.variational_optimizer.step()
                
                # Update individual Q-networks
                for i, agent in enumerate(self.agents):
                    agent_obs = episode['observations'][i][t]
                    agent_q_values = agent.get_q_values(agent_obs, episode_latent)
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
                total_mi_loss += mi_loss.item()
                total_kl_loss += kl_loss.item()
        
        return {
            'q_loss': total_q_loss / len(batch_episodes),
            'mi_loss': total_mi_loss / len(batch_episodes),
            'kl_loss': total_kl_loss / len(batch_episodes)
        }
    
    def _update_target_networks(self):
        """Update target networks."""
        self.target_mixing_network.load_state_dict(self.mixing_network.state_dict())
        for agent in self.agents:
            agent.update_target_network()
        
        print(f"Updated target networks at step {self.total_steps}")
