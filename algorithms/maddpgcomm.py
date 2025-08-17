"""
Multi-Agent Deep Deterministic Policy Gradient with Communication (MADDPG-Comm).

Extends MADDPG with explicit communication channels between agents
for improved coordination in continuous action spaces.
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


class MADDPGCommAgent(MARLAgent):
    """
    MADDPG agent with communication capabilities.
    """
    
    def __init__(self, agent_id: int, obs_space: Dict, action_space: int, config: Dict):
        """
        Initialize the MADDPG-Comm agent.
        
        Args:
            agent_id: Unique identifier for this agent
            obs_space: Individual observation space
            action_space: Number of available actions (discretized)
            config: Configuration dictionary
        """
        super().__init__(agent_id, obs_space, action_space, config)
        
        # MADDPG-Comm hyperparameters
        self.gamma = config.get('gamma', 0.99)
        self.tau = config.get('tau', 0.01)
        self.communication_dim = config.get('communication_dim', 64)
        self.noise_std = config.get('noise_std', 0.2)
        self.noise_clip = config.get('noise_clip', 0.5)
        
        # Actor network (policy) with communication
        self.actor = MADDPGCommActor(obs_space, action_space, self.communication_dim, config).to(self.device)
        self.target_actor = MADDPGCommActor(obs_space, action_space, self.communication_dim, config).to(self.device)
        
        # Critic network (centralized)
        self.critic = MADDPGCommCritic(obs_space, action_space, self.communication_dim, config).to(self.device)
        self.target_critic = MADDPGCommCritic(obs_space, action_space, self.communication_dim, config).to(self.device)
        
        # Copy weights to target networks
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = Adam(self.actor.parameters(), lr=config.get('actor_lr', 1e-4))
        self.critic_optimizer = Adam(self.critic.parameters(), lr=config.get('critic_lr', 1e-3))
        
        # Communication state
        self.last_communication = torch.zeros(self.communication_dim).to(self.device)
        self.communication_history = deque(maxlen=config.get('comm_history_length', 10))
        
        print(f"Initialized MADDPG-Comm Agent {agent_id}")
        print(f"Actor parameters: {sum(p.numel() for p in self.actor.parameters())}")
        print(f"Critic parameters: {sum(p.numel() for p in self.critic.parameters())}")
        print(f"Communication dimension: {self.communication_dim}")
    
    def get_action(self, observation: Dict, received_messages: List[torch.Tensor] = None, 
                   all_observations: List[Dict] = None, training: bool = True) -> Tuple[int, torch.Tensor, float]:
        """
        Select an action and generate communication message.
        
        Args:
            observation: Agent observation
            received_messages: Messages from other agents
            all_observations: Observations of all agents (for centralized critic)
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
        
        # Get action probabilities and communication from actor
        with torch.no_grad() if not training else torch.enable_grad():
            action_logits, communication_message = self.actor(obs_tensor, aggregated_message.unsqueeze(0))
            
            # Convert to discrete action (softmax + sampling)
            action_probs = F.softmax(action_logits, dim=-1)
            
            if training:
                # Add exploration noise
                noise = torch.randn_like(action_logits) * self.noise_std
                noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
                action_logits = action_logits + noise
                action_probs = F.softmax(action_logits, dim=-1)
                
                # Sample action
                action_dist = torch.distributions.Categorical(action_probs[0])
                action = action_dist.sample()
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
        
        # Attention-based aggregation
        if len(messages) == 1:
            return messages[0]
        
        # Simple averaging for now (can be replaced with attention mechanism)
        aggregated = torch.stack(messages).mean(dim=0)
        return aggregated
    
    def get_critic_value(self, all_observations: List[Dict], all_actions: List[int], 
                        all_messages: List[torch.Tensor]) -> torch.Tensor:
        """Get critic value for centralized training."""
        # Process all observations
        obs_tensors = []
        for obs in all_observations:
            obs_tensor = self._process_observation(obs)
            obs_tensors.append(obs_tensor.squeeze(0))
        
        all_obs_tensor = torch.stack(obs_tensors).unsqueeze(0).to(self.device)
        
        # Process all actions
        all_actions_tensor = torch.tensor(all_actions, dtype=torch.long).unsqueeze(0).to(self.device)
        
        # Process all messages
        if all_messages:
            all_messages_tensor = torch.stack(all_messages).unsqueeze(0).to(self.device)
        else:
            all_messages_tensor = torch.zeros(1, len(all_observations), self.communication_dim).to(self.device)
        
        return self.critic(all_obs_tensor, all_actions_tensor, all_messages_tensor)
    
    def update(self, batch_data: Dict) -> Dict[str, float]:
        """
        Update actor and critic networks.
        
        Args:
            batch_data: Dictionary containing transitions
            
        Returns:
            Dictionary containing loss metrics
        """
        all_observations = batch_data['all_observations']
        all_actions = batch_data['all_actions']
        rewards = batch_data['rewards']
        all_next_observations = batch_data['all_next_observations']
        dones = batch_data['dones']
        all_messages = batch_data.get('all_messages', [])
        all_next_messages = batch_data.get('all_next_messages', [])
        
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        
        for i in range(len(rewards)):
            # Update networks
            actor_loss, critic_loss = self._update_networks(
                all_observations[i], all_actions[i], rewards[i], 
                all_next_observations[i], dones[i],
                all_messages[i] if i < len(all_messages) else None,
                all_next_messages[i] if i < len(all_next_messages) else None
            )
            
            total_actor_loss += actor_loss
            total_critic_loss += critic_loss
        
        # Update target networks
        self._soft_update_targets()
        
        return {
            'actor_loss': total_actor_loss / len(rewards),
            'critic_loss': total_critic_loss / len(rewards)
        }
    
    def _update_networks(self, all_observations: List[Dict], all_actions: List[int], reward: float,
                        all_next_observations: List[Dict], done: bool,
                        all_messages: List[torch.Tensor] = None, 
                        all_next_messages: List[torch.Tensor] = None) -> Tuple[float, float]:
        """Update actor and critic networks."""
        # Process observations and actions
        obs_tensors = []
        next_obs_tensors = []
        
        for j, (obs, next_obs) in enumerate(zip(all_observations, all_next_observations)):
            obs_tensor = self._process_observation(obs)
            next_obs_tensor = self._process_observation(next_obs)
            obs_tensors.append(obs_tensor.squeeze(0))
            next_obs_tensors.append(next_obs_tensor.squeeze(0))
        
        all_obs_tensor = torch.stack(obs_tensors).unsqueeze(0).to(self.device)
        all_next_obs_tensor = torch.stack(next_obs_tensors).unsqueeze(0).to(self.device)
        all_actions_tensor = torch.tensor(all_actions, dtype=torch.long).unsqueeze(0).to(self.device)
        
        # Handle communication messages
        if all_messages:
            all_messages_tensor = torch.stack(all_messages).unsqueeze(0).to(self.device)
        else:
            all_messages_tensor = torch.zeros(1, len(all_observations), self.communication_dim).to(self.device)
        
        if all_next_messages:
            all_next_messages_tensor = torch.stack(all_next_messages).unsqueeze(0).to(self.device)
        else:
            all_next_messages_tensor = torch.zeros(1, len(all_next_observations), self.communication_dim).to(self.device)
        
        # Update Critic
        with torch.no_grad():
            # Target actions and messages from target actors
            next_action_logits_list = []
            next_messages_list = []
            
            for j in range(len(all_next_observations)):
                if j < len(all_next_messages) and all_next_messages[j] is not None:
                    received_msg = all_next_messages[j]
                else:
                    received_msg = torch.zeros(self.communication_dim).to(self.device)
                
                next_obs_single = next_obs_tensors[j].unsqueeze(0)
                next_action_logits, next_comm_msg = self.target_actor(next_obs_single, received_msg.unsqueeze(0))
                next_action = torch.argmax(F.softmax(next_action_logits, dim=-1), dim=-1)
                
                next_action_logits_list.append(next_action[0])
                next_messages_list.append(next_comm_msg[0])
            
            next_all_actions = torch.stack(next_action_logits_list).unsqueeze(0).to(self.device)
            next_all_messages = torch.stack(next_messages_list).unsqueeze(0).to(self.device)
            
            # Target Q-value
            target_q = self.target_critic(all_next_obs_tensor, next_all_actions, next_all_messages)
            target_q = reward + (1 - done) * self.gamma * target_q
        
        # Current Q-value
        current_q = self.critic(all_obs_tensor, all_actions_tensor, all_messages_tensor)
        
        # Critic loss
        critic_loss = F.mse_loss(current_q, target_q)
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update Actor
        # Get actions for this agent
        my_obs = obs_tensors[self.agent_id].unsqueeze(0)
        if self.agent_id < len(all_messages) and all_messages[self.agent_id] is not None:
            my_received_msg = all_messages[self.agent_id]
        else:
            my_received_msg = torch.zeros(self.communication_dim).to(self.device)
        
        action_logits, comm_msg = self.actor(my_obs, my_received_msg.unsqueeze(0))
        predicted_action = torch.argmax(F.softmax(action_logits, dim=-1), dim=-1)
        
        # Create new action tensor with predicted action
        new_actions = all_actions_tensor.clone()
        new_actions[0, self.agent_id] = predicted_action[0]
        
        # Create new messages tensor with predicted message
        new_messages = all_messages_tensor.clone()
        new_messages[0, self.agent_id] = comm_msg[0]
        
        # Actor loss (policy gradient)
        actor_q = self.critic(all_obs_tensor, new_actions, new_messages)
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
        torch.save(checkpoint, f"{path}_maddpgcomm_agent_{self.agent_id}.pth")
        print(f"Saved MADDPG-Comm Agent {self.agent_id} model to {path}_maddpgcomm_agent_{self.agent_id}.pth")
    
    def load_model(self, path: str):
        """Load the agent's model."""
        checkpoint = torch.load(f"{path}_maddpgcomm_agent_{self.agent_id}.pth", map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.target_actor.load_state_dict(checkpoint['target_actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.last_communication = checkpoint['last_communication']
        self.communication_history = deque(checkpoint['communication_history'], 
                                         maxlen=self.config.get('comm_history_length', 10))
        
        print(f"Loaded MADDPG-Comm Agent {self.agent_id} model from {path}_maddpgcomm_agent_{self.agent_id}.pth")


class MADDPGCommActor(nn.Module):
    """Actor network with communication generation for MADDPG-Comm."""
    
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
            Tuple of (action_logits, communication_message)
        """
        # Encode observation
        obs_features = self.obs_encoder(obs)
        
        # Encode received communication
        comm_features = self.comm_encoder(received_message)
        
        # Combine features
        combined_features = torch.cat([obs_features, comm_features], dim=-1)
        processed_features = self.feature_processor(combined_features)
        
        # Generate action logits
        action_logits = self.action_head(processed_features)
        
        # Generate communication message
        communication_message = self.communication_head(processed_features)
        
        return action_logits, communication_message


class MADDPGCommCritic(nn.Module):
    """Centralized critic network with communication awareness for MADDPG-Comm."""
    
    def __init__(self, obs_space: Dict, action_space: int, communication_dim: int, config: Dict):
        super().__init__()
        
        self.obs_dim = self._estimate_obs_dim(obs_space)
        self.action_space = action_space
        self.communication_dim = communication_dim
        
        hidden_dim = config.get('hidden_dim', 256)
        
        # This will be set by the algorithm based on number of agents
        self.n_agents = config.get('n_agents', 2)
        
        # State-action encoder (for all agents)
        total_obs_dim = self.obs_dim * self.n_agents
        total_action_dim = self.n_agents  # Discrete actions
        total_comm_dim = communication_dim * self.n_agents
        
        self.state_action_encoder = nn.Sequential(
            nn.Linear(total_obs_dim + total_action_dim + total_comm_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
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
    
    def forward(self, all_obs: torch.Tensor, all_actions: torch.Tensor, 
               all_messages: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through centralized critic.
        
        Args:
            all_obs: All agents' observations [batch_size, n_agents, obs_dim]
            all_actions: All agents' actions [batch_size, n_agents]
            all_messages: All agents' communication messages [batch_size, n_agents, comm_dim]
            
        Returns:
            Q-value for the joint action
        """
        batch_size = all_obs.shape[0]
        
        # Flatten observations
        flat_obs = all_obs.view(batch_size, -1)
        
        # Convert actions to float and flatten
        flat_actions = all_actions.float().view(batch_size, -1)
        
        # Flatten messages
        flat_messages = all_messages.view(batch_size, -1)
        
        # Combine all inputs
        combined_input = torch.cat([flat_obs, flat_actions, flat_messages], dim=-1)
        
        # Get Q-value
        q_value = self.state_action_encoder(combined_input)
        
        return q_value


class MADDPGCommReplayBuffer:
    """Replay buffer for MADDPG-Comm with communication messages."""
    
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
            'all_observations': [t['all_observations'] for t in batch],
            'all_actions': [t['all_actions'] for t in batch],
            'rewards': [t['reward'] for t in batch],
            'all_next_observations': [t['all_next_observations'] for t in batch],
            'dones': [t['done'] for t in batch],
            'all_messages': [t.get('all_messages') for t in batch],
            'all_next_messages': [t.get('all_next_messages') for t in batch]
        }
        
        return batch_data
    
    def __len__(self):
        return len(self.buffer)


class MADDPGComm(MARLAlgorithm):
    """
    Multi-Agent Deep Deterministic Policy Gradient with Communication.
    
    Extends MADDPG with explicit communication channels between agents
    for improved coordination.
    """
    
    def __init__(self, env, config: Dict, device: torch.device):
        """
        Initialize MADDPG-Comm algorithm.
        
        Args:
            env: Multi-agent environment
            config: Configuration dictionary
            device: PyTorch device
        """
        super().__init__(env, config, device)
        
        # MADDPG-Comm specific parameters
        self.batch_size = config.get('batch_size', 32)
        self.memory_size = config.get('memory_size', 10000)
        self.train_start = config.get('train_start', 1000)
        self.update_interval = config.get('update_interval', 1)
        
        # Communication parameters
        self.communication_range = config.get('communication_range', float('inf'))
        
        # Set number of agents in config for critic
        self.config['n_agents'] = self.n_agents
        
        # Create shared replay buffer
        self.replay_buffer = MADDPGCommReplayBuffer(self.memory_size)
        
        print(f"Initialized MADDPG-Comm")
        print(f"Number of agents: {self.n_agents}")
        print(f"Communication range: {self.communication_range}")
        print(f"Batch size: {self.batch_size}")
    
    def _create_agents(self):
        """Create MADDPG-Comm agents."""
        obs_space = self._get_obs_space()
        action_space = len(self.env.actions)
        
        for i in range(self.n_agents):
            agent = MADDPGCommAgent(i, obs_space, action_space, self.config)
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
        """Collect rollout with communication between agents."""
        obs = env.reset()
        done = False
        step_count = 0
        total_reward = [0.0] * self.n_agents
        
        # Initialize communication messages
        agent_messages = [torch.zeros(self.config.get('communication_dim', 64)).to(self.device) 
                         for _ in range(self.n_agents)]
        
        while not done:
            # Get all observations
            all_observations = []
            for i in range(self.n_agents):
                agent_obs = self._extract_agent_obs(obs, i)
                all_observations.append(agent_obs)
            
            # Get communication messages from previous step
            prev_messages = [agent.get_communication_message() for agent in self.agents]
            
            # Get actions and new communication messages
            actions = []
            new_messages = []
            
            for i, agent in enumerate(self.agents):
                # Get messages from other agents (excluding self)
                received_messages = [prev_messages[j] for j in range(self.n_agents) if j != i]
                
                action, comm_msg, _ = agent.get_action(
                    all_observations[i], received_messages, all_observations, training=True
                )
                actions.append(action)
                new_messages.append(comm_msg)
            
            # Take environment step
            next_obs, rewards, done, info = env.step(actions)
            
            # Get next observations
            all_next_observations = []
            for i in range(self.n_agents):
                next_agent_obs = self._extract_agent_obs(next_obs, i)
                all_next_observations.append(next_agent_obs)
            
            # Store transition
            reward = rewards[0] if isinstance(rewards, list) else rewards  # Shared reward
            for i in range(self.n_agents):
                total_reward[i] += reward
            
            transition = {
                'all_observations': all_observations,
                'all_actions': actions,
                'reward': reward,
                'all_next_observations': all_next_observations,
                'done': done,
                'all_messages': prev_messages,
                'all_next_messages': new_messages
            }
            
            self.replay_buffer.add(transition)
            
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
        """Perform MADDPG-Comm training step."""
        if self.total_steps < self.train_start:
            return {
                'episode_length': rollout_data['episode_length'],
                'total_reward': sum(rollout_data['total_reward']) / self.n_agents
            }
        
        # Train each agent if we have enough experience
        if self.total_steps % self.update_interval == 0 and len(self.replay_buffer) >= self.batch_size:
            agent_losses = []
            
            for agent in self.agents:
                # Sample batch and update agent
                batch_data = self.replay_buffer.sample(self.batch_size)
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
