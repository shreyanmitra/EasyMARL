"""
Minimax-Q Learning algorithm.

Minimax-Q extends Q-learning to multi-agent environments by computing
Q-values based on minimax optimization, assuming adversarial settings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from typing import Dict, List, Tuple, Any
from collections import deque
import random
from scipy.optimize import linprog

from .base import MARLAgent, MARLAlgorithm


class MinimaxQAgent(MARLAgent):
    """
    Minimax-Q agent that learns optimal policies in adversarial settings.
    """
    
    def __init__(self, agent_id: int, obs_space: Dict, action_space: int, config: Dict):
        """
        Initialize the Minimax-Q agent.
        
        Args:
            agent_id: Unique identifier for this agent
            obs_space: Individual observation space
            action_space: Number of available actions
            config: Configuration dictionary
        """
        super().__init__(agent_id, obs_space, action_space, config)
        
        # Minimax-Q hyperparameters
        self.gamma = config.get('gamma', 0.99)
        self.epsilon_start = config.get('epsilon_start', 1.0)
        self.epsilon_end = config.get('epsilon_end', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.epsilon = self.epsilon_start
        
        # Opponent modeling
        self.n_opponents = config.get('n_opponents', 1)
        self.opponent_action_space = action_space  # Assume same action space
        
        # Q-network for minimax values
        self.q_network = MinimaxQNetwork(obs_space, action_space, self.opponent_action_space, config).to(self.device)
        self.target_q_network = MinimaxQNetwork(obs_space, action_space, self.opponent_action_space, config).to(self.device)
        
        # Copy weights to target network
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = Adam(self.q_network.parameters(), lr=config.get('learning_rate', 1e-3))
        
        # Store recent opponent actions for modeling
        self.opponent_action_history = deque(maxlen=config.get('history_length', 1000))
        
        print(f"Initialized Minimax-Q Agent {agent_id}")
        print(f"Q-network parameters: {sum(p.numel() for p in self.q_network.parameters())}")
        print(f"Opponent action space: {self.opponent_action_space}")
    
    def get_action(self, observation: Dict, opponent_history: List = None, training: bool = True) -> Tuple[int, float]:
        """
        Select an action using minimax strategy.
        
        Args:
            observation: Agent observation
            opponent_history: History of opponent actions
            training: Whether in training mode
            
        Returns:
            Tuple of (action, q_value)
        """
        with torch.no_grad():
            if training and random.random() < self.epsilon:
                action = random.randint(0, self.action_space - 1)
                q_value = 0.0
            else:
                # Get minimax action
                action, q_value = self._get_minimax_action(observation)
            
            return action, q_value
    
    def _get_minimax_action(self, observation: Dict) -> Tuple[int, float]:
        """Compute minimax action using linear programming."""
        obs_tensor = self._process_observation(observation)
        
        # Get Q-values for all joint actions
        q_values = self.q_network(obs_tensor)[0]  # [action_space, opponent_action_space]
        
        # Solve minimax optimization using linear programming
        action, value = self._solve_minimax(q_values.cpu().numpy())
        
        return action, value
    
    def _solve_minimax(self, q_matrix: np.ndarray) -> Tuple[int, float]:
        """
        Solve minimax optimization problem using linear programming.
        
        Args:
            q_matrix: Q-values matrix [action_space, opponent_action_space]
            
        Returns:
            Tuple of (optimal_action, minimax_value)
        """
        num_actions = q_matrix.shape[0]
        num_opponent_actions = q_matrix.shape[1]
        
        try:
            # Variables: [v, p_1, p_2, ..., p_n] where v is the value and p_i are probabilities
            c = np.zeros(num_actions + 1)
            c[0] = -1  # Maximize v (minimize -v)
            
            # Constraints: sum(p_i * Q(i, o)) >= v for each opponent action o
            A_ub = []
            b_ub = []
            
            for o in range(num_opponent_actions):
                constraint = np.zeros(num_actions + 1)
                constraint[0] = 1  # v
                for a in range(num_actions):
                    constraint[a + 1] = -q_matrix[a, o]  # -p_a * Q(a, o)
                A_ub.append(constraint)
                b_ub.append(0)
            
            # Equality constraint: sum of probabilities = 1
            A_eq = np.zeros((1, num_actions + 1))
            A_eq[0, 1:] = 1  # sum of p_i = 1
            b_eq = np.array([1])
            
            # Bounds: v is free, probabilities >= 0
            bounds = [(None, None)] + [(0, None)] * num_actions
            
            # Solve linear program
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                           bounds=bounds, method='highs')
            
            if result.success:
                # Extract mixed strategy
                probabilities = result.x[1:]
                minimax_value = -result.fun
                
                # Sample action according to mixed strategy
                action = np.random.choice(num_actions, p=probabilities)
                
                return action, minimax_value
            else:
                # Fallback to uniform random if LP fails
                action = np.random.randint(num_actions)
                value = np.mean(q_matrix[action, :])
                return action, value
                
        except Exception as e:
            print(f"Minimax optimization failed: {e}")
            # Fallback to best response against uniform opponent
            action = np.argmax(np.mean(q_matrix, axis=1))
            value = np.mean(q_matrix[action, :])
            return action, value
    
    def update(self, batch_data: Dict) -> Dict[str, float]:
        """
        Update Q-network using minimax Q-learning.
        
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
        opponent_actions = batch_data.get('opponent_actions', [])
        
        total_loss = 0.0
        
        for i in range(len(observations)):
            # Update with minimax target
            loss = self._update_q_network(
                observations[i], actions[i], rewards[i], 
                next_observations[i], dones[i],
                opponent_actions[i] if i < len(opponent_actions) else None
            )
            total_loss += loss
        
        return {
            'q_loss': total_loss / len(observations),
            'epsilon': self.epsilon
        }
    
    def _update_q_network(self, observation: Dict, action: int, reward: float,
                         next_observation: Dict, done: bool, opponent_action: int = None) -> float:
        """Update Q-network with minimax target."""
        obs_tensor = self._process_observation(observation)
        next_obs_tensor = self._process_observation(next_observation)
        
        # Current Q-values
        current_q_matrix = self.q_network(obs_tensor)[0]  # [action_space, opponent_action_space]
        
        if opponent_action is not None:
            current_q = current_q_matrix[action, opponent_action]
        else:
            # If no opponent action, use expected value over all opponent actions
            current_q = torch.mean(current_q_matrix[action, :])
        
        # Target Q-value using minimax
        with torch.no_grad():
            if not done:
                next_q_matrix = self.target_q_network(next_obs_tensor)[0]
                
                # Compute minimax value for next state
                next_minimax_value = self._compute_minimax_value(next_q_matrix.cpu().numpy())
                target = reward + self.gamma * next_minimax_value
            else:
                target = reward
            
            target = torch.tensor(target, dtype=torch.float32).to(self.device)
        
        # Minimax Q-learning loss
        loss = F.mse_loss(current_q, target)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def _compute_minimax_value(self, q_matrix: np.ndarray) -> float:
        """Compute minimax value for a Q-matrix."""
        try:
            _, minimax_value = self._solve_minimax(q_matrix)
            return minimax_value
        except:
            # Fallback to max-min value
            return np.max(np.min(q_matrix, axis=1))
    
    def update_opponent_model(self, opponent_action: int):
        """Update opponent action history for modeling."""
        self.opponent_action_history.append(opponent_action)
    
    def get_opponent_distribution(self) -> np.ndarray:
        """Get empirical distribution of opponent actions."""
        if len(self.opponent_action_history) == 0:
            return np.ones(self.opponent_action_space) / self.opponent_action_space
        
        distribution = np.zeros(self.opponent_action_space)
        for action in self.opponent_action_history:
            distribution[action] += 1
        
        return distribution / len(self.opponent_action_history)
    
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
            'opponent_action_history': list(self.opponent_action_history),
            'agent_id': self.agent_id,
            'config': self.config
        }
        torch.save(checkpoint, f"{path}_minimaxq_agent_{self.agent_id}.pth")
        print(f"Saved Minimax-Q Agent {self.agent_id} model to {path}_minimaxq_agent_{self.agent_id}.pth")
    
    def load_model(self, path: str):
        """Load the agent's model."""
        checkpoint = torch.load(f"{path}_minimaxq_agent_{self.agent_id}.pth", map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_q_network.load_state_dict(checkpoint['target_q_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.opponent_action_history = deque(checkpoint['opponent_action_history'], 
                                           maxlen=self.config.get('history_length', 1000))
        
        print(f"Loaded Minimax-Q Agent {self.agent_id} model from {path}_minimaxq_agent_{self.agent_id}.pth")


class MinimaxQNetwork(nn.Module):
    """Q-network for Minimax-Q learning."""
    
    def __init__(self, obs_space: Dict, action_space: int, opponent_action_space: int, config: Dict):
        super().__init__()
        
        self.obs_dim = self._estimate_obs_dim(obs_space)
        self.action_space = action_space
        self.opponent_action_space = opponent_action_space
        
        hidden_dim = config.get('hidden_dim', 128)
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Q-value head for joint actions
        self.q_head = nn.Linear(hidden_dim, action_space * opponent_action_space)
        
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
        """
        Forward pass through minimax Q-network.
        
        Args:
            obs: Observation tensor
            
        Returns:
            Q-values for all joint actions [batch_size, action_space, opponent_action_space]
        """
        # Extract features
        features = self.feature_extractor(obs)
        
        # Get Q-values
        q_values = self.q_head(features)
        
        # Reshape to joint action space
        batch_size = obs.shape[0]
        q_values = q_values.view(batch_size, self.action_space, self.opponent_action_space)
        
        return q_values


class MinimaxQReplayBuffer:
    """Replay buffer for Minimax-Q with opponent action information."""
    
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
            'opponent_actions': [t.get('opponent_action') for t in batch]
        }
        
        return batch_data
    
    def __len__(self):
        return len(self.buffer)


class MinimaxQ(MARLAlgorithm):
    """
    Minimax-Q Learning algorithm.
    
    Extends Q-learning to multi-agent environments using minimax optimization
    for adversarial settings.
    """
    
    def __init__(self, env, config: Dict, device: torch.device):
        """
        Initialize Minimax-Q algorithm.
        
        Args:
            env: Multi-agent environment
            config: Configuration dictionary
            device: PyTorch device
        """
        super().__init__(env, config, device)
        
        # Minimax-Q specific parameters
        self.batch_size = config.get('batch_size', 32)
        self.memory_size = config.get('memory_size', 10000)
        self.train_start = config.get('train_start', 1000)
        self.update_interval = config.get('update_interval', 1)
        self.target_update_freq = config.get('target_update_freq', 100)
        
        # Create replay buffers for each agent
        self.replay_buffers = [
            MinimaxQReplayBuffer(self.memory_size) for _ in range(self.n_agents)
        ]
        
        print(f"Initialized Minimax-Q")
        print(f"Number of agents: {self.n_agents}")
        print(f"Batch size: {self.batch_size}")
    
    def _create_agents(self):
        """Create Minimax-Q agents."""
        obs_space = self._get_obs_space()
        action_space = len(self.env.actions)
        
        # Add opponent information to config
        config_with_opponents = self.config.copy()
        config_with_opponents['n_opponents'] = self.n_agents - 1
        
        for i in range(self.n_agents):
            agent = MinimaxQAgent(i, obs_space, action_space, config_with_opponents)
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
        """Collect rollout data with opponent action modeling."""
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
            
            # Store transitions with opponent actions
            for i in range(self.n_agents):
                agent_obs = self._extract_agent_obs(obs, i)
                next_agent_obs = self._extract_agent_obs(next_obs, i)
                
                reward = rewards[i] if isinstance(rewards, list) else rewards
                total_reward[i] += reward
                
                # Get opponent action (simplified: use first opponent)
                opponent_action = actions[(i + 1) % self.n_agents] if self.n_agents > 1 else None
                
                transition = {
                    'observation': agent_obs,
                    'action': actions[i],
                    'reward': reward,
                    'next_observation': next_agent_obs,
                    'done': done,
                    'opponent_action': opponent_action
                }
                
                self.replay_buffers[i].add(transition)
                
                # Update opponent model
                if opponent_action is not None:
                    self.agents[i].update_opponent_model(opponent_action)
            
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
        """Perform Minimax-Q training step."""
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
                    'total_steps': self.total_steps
                })
                
                return avg_metrics
        
        return {
            'episode_length': rollout_data['episode_length'],
            'total_reward': sum(rollout_data['total_reward']) / self.n_agents,
            'total_steps': self.total_steps
        }
