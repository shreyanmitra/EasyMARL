"""
Neural Fictitious Self-Play (NFSP) algorithm.

NFSP combines reinforcement learning with supervised learning to approximate
Nash equilibria in multi-agent games through self-play.
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


class NFSPAgent(MARLAgent):
    """
    NFSP agent that learns through neural fictitious self-play.
    """
    
    def __init__(self, agent_id: int, obs_space: Dict, action_space: int, config: Dict):
        """
        Initialize the NFSP agent.
        
        Args:
            agent_id: Unique identifier for this agent
            obs_space: Individual observation space
            action_space: Number of available actions
            config: Configuration dictionary
        """
        super().__init__(agent_id, obs_space, action_space, config)
        
        # NFSP hyperparameters
        self.gamma = config.get('gamma', 0.99)
        self.epsilon_start = config.get('epsilon_start', 0.06)
        self.epsilon_end = config.get('epsilon_end', 0.001)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.epsilon = self.epsilon_start
        
        # NFSP specific parameters
        self.eta = config.get('eta', 0.1)  # Probability of best response vs average strategy
        self.anticipatory_param = config.get('anticipatory_param', 0.1)
        
        # Q-network for best response (DQN)
        self.q_network = NFSPQNetwork(obs_space, action_space, config).to(self.device)
        self.target_q_network = NFSPQNetwork(obs_space, action_space, config).to(self.device)
        
        # Average strategy network (supervised learning)
        self.average_strategy_network = NFSPStrategyNetwork(obs_space, action_space, config).to(self.device)
        
        # Copy weights to target network
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizers
        self.q_optimizer = Adam(self.q_network.parameters(), lr=config.get('q_lr', 1e-3))
        self.strategy_optimizer = Adam(self.average_strategy_network.parameters(), 
                                     lr=config.get('strategy_lr', 1e-3))
        
        # Replay buffers
        self.rl_memory_size = config.get('rl_memory_size', 100000)
        self.sl_memory_size = config.get('sl_memory_size', 1000000)
        self.rl_buffer = deque(maxlen=self.rl_memory_size)
        self.sl_buffer = deque(maxlen=self.sl_memory_size)
        
        # Strategy tracking
        self.strategy_update_freq = config.get('strategy_update_freq', 128)
        self.step_count = 0
        
        print(f"Initialized NFSP Agent {agent_id}")
        print(f"Q-network parameters: {sum(p.numel() for p in self.q_network.parameters())}")
        print(f"Strategy network parameters: {sum(p.numel() for p in self.average_strategy_network.parameters())}")
    
    def get_action(self, observation: Dict, mode: str = 'average', training: bool = True) -> Tuple[int, float]:
        """
        Select an action using either best response or average strategy.
        
        Args:
            observation: Agent observation
            mode: 'best_response' or 'average' strategy
            training: Whether in training mode
            
        Returns:
            Tuple of (action, action_value)
        """
        obs_tensor = self._process_observation(observation)
        
        with torch.no_grad():
            if mode == 'best_response':
                # Use epsilon-greedy with Q-network
                if training and random.random() < self.epsilon:
                    action = random.randint(0, self.action_space - 1)
                    action_value = 0.0
                else:
                    q_values = self.q_network(obs_tensor)[0]
                    action = torch.argmax(q_values).item()
                    action_value = q_values[action].item()
            else:
                # Use average strategy network
                action_probs = self.average_strategy_network(obs_tensor)[0]
                
                if training:
                    # Sample from policy
                    action_dist = torch.distributions.Categorical(action_probs)
                    action = action_dist.sample().item()
                else:
                    # Greedy action
                    action = torch.argmax(action_probs).item()
                
                action_value = action_probs[action].item()
        
        return action, action_value
    
    def select_action_mode(self) -> str:
        """Select between best response and average strategy."""
        return 'best_response' if random.random() < self.eta else 'average'
    
    def add_rl_transition(self, transition: Dict):
        """Add transition to RL buffer."""
        self.rl_buffer.append(transition)
    
    def add_sl_transition(self, observation: Dict, action: int):
        """Add state-action pair to supervised learning buffer."""
        obs_tensor = self._process_observation(observation)
        self.sl_buffer.append({
            'observation': obs_tensor.squeeze(0),
            'action': action
        })
    
    def update_q_network(self, batch_size: int = 32) -> float:
        """Update Q-network using DQN."""
        if len(self.rl_buffer) < batch_size:
            return 0.0
        
        # Sample batch from RL buffer
        batch = random.sample(self.rl_buffer, batch_size)
        
        observations = torch.stack([t['observation'] for t in batch]).to(self.device)
        actions = torch.tensor([t['action'] for t in batch], dtype=torch.long).to(self.device)
        rewards = torch.tensor([t['reward'] for t in batch], dtype=torch.float32).to(self.device)
        next_observations = torch.stack([t['next_observation'] for t in batch]).to(self.device)
        dones = torch.tensor([t['done'] for t in batch], dtype=torch.bool).to(self.device)
        
        # Current Q-values
        current_q_values = self.q_network(observations).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q-values
        with torch.no_grad():
            next_q_values = self.target_q_network(next_observations).max(1)[0]
            target_q_values = rewards + (1 - dones.float()) * self.gamma * next_q_values
        
        # Q-learning loss
        q_loss = F.mse_loss(current_q_values, target_q_values)
        
        # Update Q-network
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        
        return q_loss.item()
    
    def update_strategy_network(self, batch_size: int = 128) -> float:
        """Update average strategy network using supervised learning."""
        if len(self.sl_buffer) < batch_size:
            return 0.0
        
        # Sample batch from SL buffer
        batch = random.sample(self.sl_buffer, batch_size)
        
        observations = torch.stack([t['observation'] for t in batch]).to(self.device)
        actions = torch.tensor([t['action'] for t in batch], dtype=torch.long).to(self.device)
        
        # Get predicted action probabilities
        action_probs = self.average_strategy_network(observations)
        
        # Cross-entropy loss for supervised learning
        strategy_loss = F.cross_entropy(action_probs, actions)
        
        # Update strategy network
        self.strategy_optimizer.zero_grad()
        strategy_loss.backward()
        self.strategy_optimizer.step()
        
        return strategy_loss.item()
    
    def update_target_network(self):
        """Update target Q-network."""
        self.target_q_network.load_state_dict(self.q_network.state_dict())
    
    def decay_epsilon(self):
        """Decay epsilon for exploration."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
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
    
    def save_model(self, path: str):
        """Save the agent's model."""
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_q_network_state_dict': self.target_q_network.state_dict(),
            'average_strategy_network_state_dict': self.average_strategy_network.state_dict(),
            'q_optimizer_state_dict': self.q_optimizer.state_dict(),
            'strategy_optimizer_state_dict': self.strategy_optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'agent_id': self.agent_id,
            'config': self.config
        }
        torch.save(checkpoint, f"{path}_nfsp_agent_{self.agent_id}.pth")
        print(f"Saved NFSP Agent {self.agent_id} model to {path}_nfsp_agent_{self.agent_id}.pth")
    
    def load_model(self, path: str):
        """Load the agent's model."""
        checkpoint = torch.load(f"{path}_nfsp_agent_{self.agent_id}.pth", map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_q_network.load_state_dict(checkpoint['target_q_network_state_dict'])
        self.average_strategy_network.load_state_dict(checkpoint['average_strategy_network_state_dict'])
        self.q_optimizer.load_state_dict(checkpoint['q_optimizer_state_dict'])
        self.strategy_optimizer.load_state_dict(checkpoint['strategy_optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.step_count = checkpoint['step_count']
        
        print(f"Loaded NFSP Agent {self.agent_id} model from {path}_nfsp_agent_{self.agent_id}.pth")


class NFSPQNetwork(nn.Module):
    """Q-network for NFSP best response."""
    
    def __init__(self, obs_space: Dict, action_space: int, config: Dict):
        super().__init__()
        
        self.obs_dim = self._estimate_obs_dim(obs_space)
        hidden_dim = config.get('hidden_dim', 256)
        
        self.network = nn.Sequential(
            nn.Linear(self.obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
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


class NFSPStrategyNetwork(nn.Module):
    """Strategy network for NFSP average strategy."""
    
    def __init__(self, obs_space: Dict, action_space: int, config: Dict):
        super().__init__()
        
        self.obs_dim = self._estimate_obs_dim(obs_space)
        hidden_dim = config.get('hidden_dim', 256)
        
        self.network = nn.Sequential(
            nn.Linear(self.obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_space),
            nn.Softmax(dim=-1)
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
        """Forward pass through strategy network."""
        return self.network(obs)


class NFSP(MARLAlgorithm):
    """
    Neural Fictitious Self-Play (NFSP) algorithm.
    
    Combines reinforcement learning with supervised learning to approximate
    Nash equilibria in multi-agent games.
    """
    
    def __init__(self, env, config: Dict, device: torch.device):
        """
        Initialize NFSP algorithm.
        
        Args:
            env: Multi-agent environment
            config: Configuration dictionary
            device: PyTorch device
        """
        super().__init__(env, config, device)
        
        # NFSP specific parameters
        self.train_start = config.get('train_start', 1000)
        self.update_interval = config.get('update_interval', 1)
        self.target_update_freq = config.get('target_update_freq', 1000)
        self.strategy_update_freq = config.get('strategy_update_freq', 128)
        
        print(f"Initialized NFSP")
        print(f"Number of agents: {self.n_agents}")
        print(f"Target update frequency: {self.target_update_freq}")
        print(f"Strategy update frequency: {self.strategy_update_freq}")
    
    def _create_agents(self):
        """Create NFSP agents."""
        obs_space = self._get_obs_space()
        action_space = len(self.env.actions)
        
        for i in range(self.n_agents):
            agent = NFSPAgent(i, obs_space, action_space, self.config)
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
        """Collect rollout with NFSP mode selection."""
        obs = env.reset()
        done = False
        step_count = 0
        total_reward = [0.0] * self.n_agents
        
        # Select action modes for each agent
        agent_modes = [agent.select_action_mode() for agent in self.agents]
        
        while not done:
            # Get actions from all agents
            actions = []
            for i, agent in enumerate(self.agents):
                agent_obs = self._extract_agent_obs(obs, i)
                action, _ = agent.get_action(agent_obs, mode=agent_modes[i], training=True)
                actions.append(action)
                
                # Add to SL buffer if using best response
                if agent_modes[i] == 'best_response':
                    agent.add_sl_transition(agent_obs, action)
            
            # Take environment step
            next_obs, rewards, done, info = env.step(actions)
            
            # Store RL transitions
            for i in range(self.n_agents):
                if agent_modes[i] == 'best_response':
                    agent_obs = self._extract_agent_obs(obs, i)
                    next_agent_obs = self._extract_agent_obs(next_obs, i)
                    
                    reward = rewards[i] if isinstance(rewards, list) else rewards
                    total_reward[i] += reward
                    
                    obs_tensor = self.agents[i]._process_observation(agent_obs)
                    next_obs_tensor = self.agents[i]._process_observation(next_agent_obs)
                    
                    transition = {
                        'observation': obs_tensor.squeeze(0),
                        'action': actions[i],
                        'reward': reward,
                        'next_observation': next_obs_tensor.squeeze(0),
                        'done': done
                    }
                    
                    self.agents[i].add_rl_transition(transition)
            
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
        """Perform NFSP training step."""
        if self.total_steps < self.train_start:
            return {
                'episode_length': rollout_data['episode_length'],
                'total_reward': sum(rollout_data['total_reward']) / self.n_agents
            }
        
        # Train each agent
        if self.total_steps % self.update_interval == 0:
            q_losses = []
            strategy_losses = []
            
            for agent in self.agents:
                # Update Q-network
                q_loss = agent.update_q_network()
                q_losses.append(q_loss)
                
                # Update strategy network
                if self.total_steps % self.strategy_update_freq == 0:
                    strategy_loss = agent.update_strategy_network()
                    strategy_losses.append(strategy_loss)
            
            # Update target networks
            if self.total_steps % self.target_update_freq == 0:
                for agent in self.agents:
                    agent.update_target_network()
                print(f"Updated target networks at step {self.total_steps}")
            
            metrics = {
                'avg_q_loss': np.mean(q_losses) if q_losses else 0.0,
                'episode_length': rollout_data['episode_length'],
                'total_reward': sum(rollout_data['total_reward']) / self.n_agents,
                'total_steps': self.total_steps
            }
            
            if strategy_losses:
                metrics['avg_strategy_loss'] = np.mean(strategy_losses)
            
            return metrics
        
        return {
            'episode_length': rollout_data['episode_length'],
            'total_reward': sum(rollout_data['total_reward']) / self.n_agents,
            'total_steps': self.total_steps
        }
