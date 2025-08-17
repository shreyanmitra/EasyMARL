"""
Independent Q-Learning (IQL) algorithm for multi-agent environments.

IQL is the simplest multi-agent Q-learning approach where each agent learns
independently using standard Q-learning, treating other agents as part of the environment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from typing import Dict, List, Tuple, Any
import copy

from .base import MARLAgent, MARLAlgorithm, ReplayBuffer


class IQLAgent(MARLAgent):
    """
    Independent Q-Learning agent implementation.
    """
    
    def __init__(self, agent_id: int, obs_space: Dict, action_space: int, config: Dict):
        """
        Initialize the IQL agent.
        
        Args:
            agent_id: Unique identifier for this agent
            obs_space: Observation space specification
            action_space: Number of available actions
            config: Configuration dictionary containing hyperparameters
        """
        super().__init__(agent_id, obs_space, action_space, config)
        
        # Q-learning hyperparameters
        self.gamma = config.get('gamma', 0.99)
        self.epsilon = config.get('epsilon_start', 1.0)
        self.epsilon_end = config.get('epsilon_end', 0.05)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.lr = config.get('lr', 1e-3)
        self.target_update_freq = config.get('target_update_freq', 100)
        
        # Networks
        self.q_network = IQLNetwork(obs_space, action_space, config).to(self.device)
        self.target_q_network = copy.deepcopy(self.q_network)
        
        # Optimizer
        self.optimizer = Adam(self.q_network.parameters(), lr=self.lr)
        
        # Training statistics
        self.update_count = 0
        
        print(f"Initialized IQL Agent {agent_id} with {sum(p.numel() for p in self.q_network.parameters())} parameters")
    
    def get_action(self, observation: Dict, training: bool = True) -> int:
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            observation: Current observation from the environment
            training: Whether the agent is in training mode
            
        Returns:
            Selected action (integer)
        """
        if training and np.random.random() < self.epsilon:
            # Random action for exploration
            return np.random.randint(0, self.action_space)
        else:
            # Greedy action
            with torch.no_grad():
                obs_tensor = self._process_observation(observation)
                q_values = self.q_network(obs_tensor)
                return q_values.argmax(dim=-1).item()
    
    def get_q_values(self, observation: Dict) -> torch.Tensor:
        """Get Q-values for the given observation."""
        obs_tensor = self._process_observation(observation)
        return self.q_network(obs_tensor)
    
    def _process_observation(self, observation: Dict) -> torch.Tensor:
        """Convert observation to tensor format."""
        if isinstance(observation, dict):
            # Handle dictionary observations
            obs_list = []
            for key, value in observation.items():
                if isinstance(value, np.ndarray):
                    obs_list.append(torch.tensor(value, dtype=torch.float32).flatten())
                else:
                    obs_list.append(torch.tensor([value], dtype=torch.float32))
            obs_tensor = torch.cat(obs_list).unsqueeze(0).to(self.device)
        else:
            # Handle array observations
            obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        return obs_tensor
    
    def store_transition(self, observation: Dict, action: int, reward: float, 
                        next_observation: Dict, done: bool):
        """Store a transition in the agent's memory."""
        self.memory['observations'].append(observation)
        self.memory['actions'].append(action)
        self.memory['rewards'].append(reward)
        self.memory['next_observations'].append(next_observation)
        self.memory['dones'].append(done)
    
    def update(self, batch_size: int = 32) -> Dict[str, float]:
        """
        Update the agent's Q-network using stored experiences.
        
        Args:
            batch_size: Size of the training batch
            
        Returns:
            Dictionary containing loss values and metrics
        """
        if len(self.memory['rewards']) < batch_size:
            return {}
        
        # Sample a batch
        indices = np.random.choice(len(self.memory['rewards']), batch_size, replace=False)
        
        # Prepare batch data
        batch_obs = [self.memory['observations'][i] for i in indices]
        batch_actions = torch.tensor([self.memory['actions'][i] for i in indices], 
                                   dtype=torch.long).to(self.device)
        batch_rewards = torch.tensor([self.memory['rewards'][i] for i in indices], 
                                   dtype=torch.float32).to(self.device)
        batch_next_obs = [self.memory['next_observations'][i] for i in indices]
        batch_dones = torch.tensor([self.memory['dones'][i] for i in indices], 
                                 dtype=torch.float32).to(self.device)
        
        # Process observations
        batch_obs_processed = []
        batch_next_obs_processed = []
        
        for obs, next_obs in zip(batch_obs, batch_next_obs):
            batch_obs_processed.append(self._process_observation(obs))
            batch_next_obs_processed.append(self._process_observation(next_obs))
        
        # Stack observations
        batch_obs_tensor = torch.cat(batch_obs_processed, dim=0)
        batch_next_obs_tensor = torch.cat(batch_next_obs_processed, dim=0)
        
        # Current Q-values
        current_q_values = self.q_network(batch_obs_tensor)
        current_q_values = current_q_values.gather(1, batch_actions.unsqueeze(1)).squeeze(1)
        
        # Target Q-values
        with torch.no_grad():
            next_q_values = self.target_q_network(batch_next_obs_tensor)
            max_next_q_values = next_q_values.max(dim=1)[0]
            target_q_values = batch_rewards + self.gamma * max_next_q_values * (1 - batch_dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values, target_q_values)
        
        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())
        
        # Update epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return {
            'loss': loss.item(),
            'epsilon': self.epsilon,
            'mean_q_value': current_q_values.mean().item()
        }
    
    def reset_memory(self):
        """Reset the agent's memory."""
        self.memory = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'next_observations': [],
            'dones': []
        }
    
    def save_model(self, path: str):
        """Save the agent's model to disk."""
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_q_network_state_dict': self.target_q_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'update_count': self.update_count,
            'agent_id': self.agent_id,
            'config': self.config
        }
        torch.save(checkpoint, f"{path}_iql_agent_{self.agent_id}.pth")
        print(f"Saved IQL Agent {self.agent_id} model to {path}_iql_agent_{self.agent_id}.pth")
    
    def load_model(self, path: str):
        """Load the agent's model from disk."""
        checkpoint = torch.load(f"{path}_iql_agent_{self.agent_id}.pth", map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_q_network.load_state_dict(checkpoint['target_q_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.update_count = checkpoint['update_count']
        print(f"Loaded IQL Agent {self.agent_id} model from {path}_iql_agent_{self.agent_id}.pth")


class IQLNetwork(nn.Module):
    """Q-network for IQL agent."""
    
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
                total_dim += 1  # Scalar value
        return total_dim
    
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass through Q-network."""
        return self.network(obs)


class IQL(MARLAlgorithm):
    """
    Independent Q-Learning (IQL) algorithm.
    
    Each agent learns independently using Q-learning.
    """
    
    def __init__(self, env, config: Dict, device: torch.device):
        """
        Initialize the IQL algorithm.
        
        Args:
            env: Multi-agent environment
            config: Configuration dictionary
            device: PyTorch device
        """
        super().__init__(env, config, device)
        
        # IQL specific parameters
        self.batch_size = config.get('batch_size', 32)
        self.memory_size = config.get('memory_size', 10000)
        self.learning_starts = config.get('learning_starts', 1000)
        self.train_freq = config.get('train_freq', 4)
        
        self.steps_done = 0
        
        print(f"Initialized IQL with {self.n_agents} independent agents")
        print(f"Batch size: {self.batch_size}")
        print(f"Memory size: {self.memory_size}")
    
    def _create_agents(self):
        """Create IQL agents for each position in the environment."""
        obs_space = self._get_obs_space()
        action_space = len(self.env.actions)
        
        for i in range(self.n_agents):
            agent = IQLAgent(i, obs_space, action_space, self.config)
            self.agents.append(agent)
            print(f"Created IQL Agent {i}")
    
    def _get_obs_space(self) -> Dict:
        """Get observation space specification from environment."""
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
    
    def collect_rollout(self, env) -> Dict:
        """
        Collect experiences and store them in agent memories.
        """
        obs = env.reset()
        done = False
        episode_rewards = [0.0] * self.n_agents
        episode_length = 0
        
        while not done:
            # Get actions from all agents
            actions = []
            
            for i, agent in enumerate(self.agents):
                # Extract individual agent observation
                agent_obs = self._extract_agent_obs(obs, i)
                action = agent.get_action(agent_obs, training=True)
                actions.append(action)
            
            # Take environment step
            next_obs, rewards, done, info = env.step(actions)
            
            # Store transitions in agent memories
            for i, agent in enumerate(self.agents):
                agent_obs = self._extract_agent_obs(obs, i)
                next_agent_obs = self._extract_agent_obs(next_obs, i)
                agent_reward = rewards[i] if isinstance(rewards, list) else rewards
                
                agent.store_transition(agent_obs, actions[i], agent_reward, next_agent_obs, done)
                episode_rewards[i] += agent_reward
                
                # Maintain memory size limit
                if len(agent.memory['rewards']) > self.memory_size:
                    # Remove oldest transition
                    for key in agent.memory:
                        agent.memory[key].pop(0)
            
            obs = next_obs
            episode_length += 1
            self.steps_done += 1
        
        return {
            'episode_rewards': episode_rewards,
            'episode_length': episode_length,
            'total_reward': sum(episode_rewards)
        }
    
    def _extract_agent_obs(self, obs, agent_id: int) -> Dict:
        """Extract observation for a specific agent."""
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
    
    def train_step(self, rollout_data: Dict) -> Dict[str, float]:
        """
        Perform one training step using the collected rollout data.
        
        Args:
            rollout_data: Data collected from environment rollouts
            
        Returns:
            Dictionary containing training metrics
        """
        metrics = {
            'episode_reward': rollout_data['total_reward'],
            'episode_length': rollout_data['episode_length'],
            'steps_done': self.steps_done
        }
        
        # Only train if we have enough samples and it's time to train
        if self.steps_done < self.learning_starts:
            return metrics
        
        if self.steps_done % self.train_freq == 0:
            # Update each agent independently
            agent_losses = []
            agent_epsilons = []
            agent_q_values = []
            
            for agent in self.agents:
                agent_metrics = agent.update(self.batch_size)
                
                if agent_metrics:  # Only add if update happened
                    agent_losses.append(agent_metrics['loss'])
                    agent_epsilons.append(agent_metrics['epsilon'])
                    agent_q_values.append(agent_metrics['mean_q_value'])
            
            # Aggregate metrics across agents
            if agent_losses:
                metrics.update({
                    'mean_loss': np.mean(agent_losses),
                    'mean_epsilon': np.mean(agent_epsilons),
                    'mean_q_value': np.mean(agent_q_values)
                })
        
        return metrics
    
    def evaluate(self, env, num_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate the current policy.
        
        Args:
            env: Environment to evaluate on
            num_episodes: Number of episodes to run
            
        Returns:
            Dictionary containing evaluation metrics
        """
        total_rewards = []
        episode_lengths = []
        
        for episode in range(num_episodes):
            obs = env.reset()
            done = False
            episode_reward = [0.0] * self.n_agents
            episode_length = 0
            
            while not done:
                # Get actions from all agents (greedy)
                actions = []
                
                for i, agent in enumerate(self.agents):
                    agent_obs = self._extract_agent_obs(obs, i)
                    action = agent.get_action(agent_obs, training=False)
                    actions.append(action)
                
                # Take environment step
                obs, rewards, done, _ = env.step(actions)
                
                # Accumulate rewards
                if isinstance(rewards, list):
                    for i in range(self.n_agents):
                        episode_reward[i] += rewards[i]
                else:
                    for i in range(self.n_agents):
                        episode_reward[i] += rewards
                
                episode_length += 1
            
            total_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        # Compute evaluation metrics
        total_rewards = np.array(total_rewards)
        eval_metrics = {
            'eval_mean_reward': np.mean(total_rewards),
            'eval_std_reward': np.std(np.mean(total_rewards, axis=1)),
            'eval_mean_episode_length': np.mean(episode_lengths),
            'eval_min_reward': np.min(total_rewards),
            'eval_max_reward': np.max(total_rewards)
        }
        
        return eval_metrics
