"""
Multi-Agent Actor-Critic with Communication (MAACC) algorithm.

MAACC extends actor-critic methods with explicit communication channels
between agents to enable coordination and information sharing.
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


class MAACCAgent(MARLAgent):
    """
    MAACC agent with communication capabilities.
    """
    
    def __init__(self, agent_id: int, obs_space: Dict, action_space: int, config: Dict):
        """
        Initialize the MAACC agent.
        
        Args:
            agent_id: Unique identifier for this agent
            obs_space: Individual observation space
            action_space: Number of available actions
            config: Configuration dictionary
        """
        super().__init__(agent_id, obs_space, action_space, config)
        
        # MAACC hyperparameters
        self.gamma = config.get('gamma', 0.99)
        self.tau = config.get('tau', 0.005)
        self.communication_dim = config.get('communication_dim', 32)
        self.communication_range = config.get('communication_range', float('inf'))
        self.noise_std = config.get('noise_std', 0.1)
        
        # Actor network (policy)
        self.actor = MAACCActor(obs_space, action_space, self.communication_dim, config).to(self.device)
        self.target_actor = MAACCActor(obs_space, action_space, self.communication_dim, config).to(self.device)
        
        # Critic network (Q-function)
        self.critic = MAACCCritic(obs_space, action_space, self.communication_dim, config).to(self.device)
        self.target_critic = MAACCCritic(obs_space, action_space, self.communication_dim, config).to(self.device)
        
        # Copy weights to target networks
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = Adam(self.actor.parameters(), lr=config.get('actor_lr', 1e-4))
        self.critic_optimizer = Adam(self.critic.parameters(), lr=config.get('critic_lr', 1e-3))
        
        # Communication state
        self.last_communication = torch.zeros(self.communication_dim).to(self.device)
        self.communication_history = deque(maxlen=config.get('comm_history_length', 10))
        
        print(f"Initialized MAACC Agent {agent_id}")
        print(f"Actor parameters: {sum(p.numel() for p in self.actor.parameters())}")
        print(f"Critic parameters: {sum(p.numel() for p in self.critic.parameters())}")
        print(f"Communication dimension: {self.communication_dim}")
    
    def get_action(self, observation: Dict, received_messages: List[torch.Tensor] = None, 
                   training: bool = True) -> Tuple[int, torch.Tensor, float]:
        """
        Select an action and generate communication message.
        
        Args:
            observation: Agent observation
            received_messages: Messages from other agents
            training: Whether in training mode
            
        Returns:
            Tuple of (action, communication_message, action_value)
        """
        obs_tensor = self._process_observation(observation)
        
        # Process received messages
        if received_messages:
            aggregated_message = self._aggregate_messages(received_messages)
        else:
            aggregated_message = torch.zeros(self.communication_dim).to(self.device)
        
        # Get action and communication from actor
        with torch.no_grad() if not training else torch.enable_grad():
            action_probs, communication_message = self.actor(obs_tensor, aggregated_message.unsqueeze(0))
            
            if training:
                # Add exploration noise
                action_probs = action_probs + torch.randn_like(action_probs) * self.noise_std
                action_probs = F.softmax(action_probs, dim=-1)
                
                # Sample action
                action_dist = torch.distributions.Categorical(action_probs[0])
                action = action_dist.sample()
                action_value = action_probs[0, action].item()
            else:
                # Greedy action selection
                action = torch.argmax(action_probs, dim=-1)[0]
                action_value = action_probs[0, action].item()
        
        # Store communication for next step
        self.last_communication = communication_message[0].detach()
        self.communication_history.append(self.last_communication.clone())
        
        return action.item(), communication_message[0], action_value
    
    def _aggregate_messages(self, messages: List[torch.Tensor]) -> torch.Tensor:
        """Aggregate received communication messages."""
        if not messages:
            return torch.zeros(self.communication_dim).to(self.device)
        
        # Simple averaging aggregation
        aggregated = torch.stack(messages).mean(dim=0)
        return aggregated
    
    def get_critic_value(self, observation: Dict, action: int, 
                        communication_message: torch.Tensor) -> torch.Tensor:
        """Get critic value for state-action pair."""
        obs_tensor = self._process_observation(observation)
        action_tensor = torch.tensor([action], dtype=torch.long).to(self.device)
        
        return self.critic(obs_tensor, action_tensor, communication_message.unsqueeze(0))
    
    def update(self, batch_data: Dict) -> Dict[str, float]:
        """
        Update actor and critic networks.
        
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
        messages = batch_data.get('messages', [])
        next_messages = batch_data.get('next_messages', [])
        
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        
        for i in range(len(observations)):
            # Update networks
            actor_loss, critic_loss = self._update_networks(
                observations[i], actions[i], rewards[i], 
                next_observations[i], dones[i],
                messages[i] if i < len(messages) else None,
                next_messages[i] if i < len(next_messages) else None
            )
            
            total_actor_loss += actor_loss
            total_critic_loss += critic_loss
        
        # Update target networks
        self._soft_update_targets()
        
        return {
            'actor_loss': total_actor_loss / len(observations),
            'critic_loss': total_critic_loss / len(observations)
        }
    
    def _update_networks(self, observation: Dict, action: int, reward: float,
                        next_observation: Dict, done: bool,
                        message: torch.Tensor = None, next_message: torch.Tensor = None) -> Tuple[float, float]:
        """Update actor and critic networks."""
        obs_tensor = self._process_observation(observation)
        next_obs_tensor = self._process_observation(next_observation)
        action_tensor = torch.tensor([action], dtype=torch.long).to(self.device)
        
        # Handle communication messages
        if message is None:
            message = torch.zeros(self.communication_dim).to(self.device)
        if next_message is None:
            next_message = torch.zeros(self.communication_dim).to(self.device)
        
        # Update Critic
        with torch.no_grad():
            # Target action and message from target actor
            next_action_probs, next_comm_msg = self.target_actor(next_obs_tensor, next_message.unsqueeze(0))
            next_action = torch.argmax(next_action_probs, dim=-1)
            
            # Target Q-value
            target_q = self.target_critic(next_obs_tensor, next_action, next_comm_msg)
            target_q = reward + (1 - done) * self.gamma * target_q
        
        # Current Q-value
        current_q = self.critic(obs_tensor, action_tensor, message.unsqueeze(0))
        
        # Critic loss
        critic_loss = F.mse_loss(current_q, target_q)
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update Actor
        action_probs, comm_msg = self.actor(obs_tensor, message.unsqueeze(0))
        predicted_action = torch.argmax(action_probs, dim=-1)
        
        # Actor loss (policy gradient)
        actor_q = self.critic(obs_tensor, predicted_action, comm_msg)
        actor_loss = -actor_q.mean()
        
        # Add communication regularization
        comm_reg = torch.norm(comm_msg, p=2, dim=-1).mean()
        actor_loss += 0.01 * comm_reg
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        return actor_loss.item(), critic_loss.item()
    
    def _soft_update_targets(self):
        """Soft update target networks."""
        for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
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
    
    def get_communication_message(self) -> torch.Tensor:
        """Get the last communication message."""
        return self.last_communication
    
    def save_model(self, path: str):
        """Save the agent's model."""
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'target_actor_state_dict': self.target_actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'target_critic_state_dict': self.target_critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'last_communication': self.last_communication,
            'communication_history': list(self.communication_history),
            'agent_id': self.agent_id,
            'config': self.config
        }
        torch.save(checkpoint, f"{path}_maacc_agent_{self.agent_id}.pth")
        print(f"Saved MAACC Agent {self.agent_id} model to {path}_maacc_agent_{self.agent_id}.pth")
    
    def load_model(self, path: str):
        """Load the agent's model."""
        checkpoint = torch.load(f"{path}_maacc_agent_{self.agent_id}.pth", map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.target_actor.load_state_dict(checkpoint['target_actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.last_communication = checkpoint['last_communication']
        self.communication_history = deque(checkpoint['communication_history'], 
                                         maxlen=self.config.get('comm_history_length', 10))
        
        print(f"Loaded MAACC Agent {self.agent_id} model from {path}_maacc_agent_{self.agent_id}.pth")


class MAACCActor(nn.Module):
    """Actor network with communication generation."""
    
    def __init__(self, obs_space: Dict, action_space: int, communication_dim: int, config: Dict):
        super().__init__()
        
        self.obs_dim = self._estimate_obs_dim(obs_space)
        self.action_space = action_space
        self.communication_dim = communication_dim
        
        hidden_dim = config.get('hidden_dim', 256)
        
        # Observation encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(self.obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Communication encoder
        self.comm_encoder = nn.Sequential(
            nn.Linear(communication_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Combined feature processor
        self.feature_processor = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Action head
        self.action_head = nn.Linear(hidden_dim, action_space)
        
        # Communication head
        self.communication_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, communication_dim),
            nn.Tanh()  # Normalize communication messages
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
    
    def forward(self, obs: torch.Tensor, received_message: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through actor network.
        
        Args:
            obs: Observation tensor
            received_message: Received communication message
            
        Returns:
            Tuple of (action_probabilities, communication_message)
        """
        # Encode observation
        obs_features = self.obs_encoder(obs)
        
        # Encode received communication
        comm_features = self.comm_encoder(received_message)
        
        # Combine features
        combined_features = torch.cat([obs_features, comm_features], dim=-1)
        processed_features = self.feature_processor(combined_features)
        
        # Generate action probabilities
        action_logits = self.action_head(processed_features)
        action_probs = F.softmax(action_logits, dim=-1)
        
        # Generate communication message
        communication_message = self.communication_head(processed_features)
        
        return action_probs, communication_message


class MAACCCritic(nn.Module):
    """Critic network with communication awareness."""
    
    def __init__(self, obs_space: Dict, action_space: int, communication_dim: int, config: Dict):
        super().__init__()
        
        self.obs_dim = self._estimate_obs_dim(obs_space)
        self.action_space = action_space
        self.communication_dim = communication_dim
        
        hidden_dim = config.get('hidden_dim', 256)
        
        # State-action encoder
        self.state_action_encoder = nn.Sequential(
            nn.Linear(self.obs_dim + 1, hidden_dim),  # +1 for action
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Communication encoder
        self.comm_encoder = nn.Sequential(
            nn.Linear(communication_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Q-value head
        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
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
    
    def forward(self, obs: torch.Tensor, action: torch.Tensor, 
               communication_message: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through critic network.
        
        Args:
            obs: Observation tensor
            action: Action tensor
            communication_message: Communication message
            
        Returns:
            Q-value
        """
        # Prepare action input
        action_one_hot = F.one_hot(action, self.action_space).float().squeeze(1)
        action_scalar = action.float().unsqueeze(-1)  # Use scalar action
        
        # Encode state-action
        state_action = torch.cat([obs, action_scalar], dim=-1)
        state_action_features = self.state_action_encoder(state_action)
        
        # Encode communication
        comm_features = self.comm_encoder(communication_message)
        
        # Combine and get Q-value
        combined_features = torch.cat([state_action_features, comm_features], dim=-1)
        q_value = self.q_head(combined_features)
        
        return q_value


class MAACCReplayBuffer:
    """Replay buffer for MAACC with communication messages."""
    
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
            'messages': [t.get('message') for t in batch],
            'next_messages': [t.get('next_message') for t in batch]
        }
        
        return batch_data
    
    def __len__(self):
        return len(self.buffer)


class MAACC(MARLAlgorithm):
    """
    Multi-Agent Actor-Critic with Communication (MAACC) algorithm.
    
    Extends actor-critic methods with explicit communication channels
    between agents for coordination.
    """
    
    def __init__(self, env, config: Dict, device: torch.device):
        """
        Initialize MAACC algorithm.
        
        Args:
            env: Multi-agent environment
            config: Configuration dictionary
            device: PyTorch device
        """
        super().__init__(env, config, device)
        
        # MAACC specific parameters
        self.batch_size = config.get('batch_size', 32)
        self.memory_size = config.get('memory_size', 10000)
        self.train_start = config.get('train_start', 1000)
        self.update_interval = config.get('update_interval', 1)
        
        # Communication parameters
        self.communication_range = config.get('communication_range', float('inf'))
        
        # Create replay buffers for each agent
        self.replay_buffers = [
            MAACCReplayBuffer(self.memory_size) for _ in range(self.n_agents)
        ]
        
        print(f"Initialized MAACC")
        print(f"Number of agents: {self.n_agents}")
        print(f"Communication range: {self.communication_range}")
        print(f"Batch size: {self.batch_size}")
    
    def _create_agents(self):
        """Create MAACC agents."""
        obs_space = self._get_obs_space()
        action_space = len(self.env.actions)
        
        for i in range(self.n_agents):
            agent = MAACCAgent(i, obs_space, action_space, self.config)
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
    
    def _get_communication_neighbors(self, agent_id: int) -> List[int]:
        """Get list of agents within communication range."""
        # For simplicity, assume all agents can communicate
        # In practice, this would depend on spatial positions
        neighbors = [i for i in range(self.n_agents) if i != agent_id]
        return neighbors
    
    def collect_rollout(self, env) -> Dict:
        """Collect rollout with communication between agents."""
        obs = env.reset()
        done = False
        step_count = 0
        total_reward = [0.0] * self.n_agents
        
        # Initialize communication messages
        agent_messages = [torch.zeros(self.config.get('communication_dim', 32)).to(self.device) 
                         for _ in range(self.n_agents)]
        
        while not done:
            # Get communication messages from previous step
            prev_messages = [agent.get_communication_message() for agent in self.agents]
            
            # Get actions and new communication messages
            actions = []
            new_messages = []
            
            for i, agent in enumerate(self.agents):
                agent_obs = self._extract_agent_obs(obs, i)
                
                # Get messages from neighbors
                neighbors = self._get_communication_neighbors(i)
                received_messages = [prev_messages[j] for j in neighbors]
                
                action, comm_msg, _ = agent.get_action(agent_obs, received_messages, training=True)
                actions.append(action)
                new_messages.append(comm_msg)
            
            # Take environment step
            next_obs, rewards, done, info = env.step(actions)
            
            # Store transitions with communication
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
                    'message': prev_messages[i],
                    'next_message': new_messages[i]
                }
                
                self.replay_buffers[i].add(transition)
            
            # Update communication messages
            agent_messages = new_messages
            obs = next_obs
            step_count += 1
            self.total_steps += 1
        
        return {
            'episode_length': step_count,
            'total_reward': total_reward
        }
    
    def train_step(self, rollout_data: Dict) -> Dict[str, float]:
        """Perform MAACC training step."""
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
