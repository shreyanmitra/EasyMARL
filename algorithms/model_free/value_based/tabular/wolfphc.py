"""
WoLF-PHC (Win or Learn Fast - Policy Hill Climbing) algorithm.

WoLF-PHC uses adaptive learning rates based on whether the agent is
"winning" or "losing" to accelerate convergence in multi-agent settings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from typing import Dict, List, Tuple, Any
from collections import deque, defaultdict
import random

from .base import MARLAgent, MARLAlgorithm


class WoLFPHCAgent(MARLAgent):
    """
    WoLF-PHC agent with adaptive learning rates.
    """
    
    def __init__(self, agent_id: int, obs_space: Dict, action_space: int, config: Dict):
        """
        Initialize the WoLF-PHC agent.
        
        Args:
            agent_id: Unique identifier for this agent
            obs_space: Individual observation space
            action_space: Number of available actions
            config: Configuration dictionary
        """
        super().__init__(agent_id, obs_space, action_space, config)
        
        # WoLF-PHC hyperparameters
        self.gamma = config.get('gamma', 0.99)
        self.learning_rate = config.get('learning_rate', 0.1)
        
        # WoLF parameters
        self.delta_win = config.get('delta_win', 0.001)   # Learning rate when winning
        self.delta_lose = config.get('delta_lose', 0.01)  # Learning rate when losing
        self.exploration_rate = config.get('exploration_rate', 0.1)
        
        # Initialize Q-values and policy tables
        self.q_table = defaultdict(lambda: np.zeros(action_space))
        self.policy = defaultdict(lambda: np.ones(action_space) / action_space)
        self.average_policy = defaultdict(lambda: np.ones(action_space) / action_space)
        self.state_visits = defaultdict(int)
        
        # Networks for function approximation (optional)
        self.use_function_approximation = config.get('use_function_approximation', False)
        if self.use_function_approximation:
            self.q_network = WoLFPHCQNetwork(obs_space, action_space, config).to(self.device)
            self.policy_network = WoLFPHCPolicyNetwork(obs_space, action_space, config).to(self.device)
        
        print(f"Initialized WoLF-PHC Agent {agent_id}")
        print(f"Delta win: {self.delta_win}, Delta lose: {self.delta_lose}")
        print(f"Function approximation: {self.use_function_approximation}")
    
    def get_action(self, observation: Dict, training: bool = True) -> Tuple[int, float]:
        """
        Select an action using current policy.
        
        Args:
            observation: Individual agent observation
            training: Whether in training mode
            
        Returns:
            Tuple of (action, action_probability)
        """
        state_key = self._get_state_key(observation)
        
        if self.use_function_approximation:
            with torch.no_grad():
                obs_tensor = self._process_observation(observation)
                policy_probs = F.softmax(self.policy_network(obs_tensor), dim=-1)
                policy_probs = policy_probs.cpu().numpy().flatten()
        else:
            policy_probs = self.policy[state_key]
        
        if training:
            # Sample action according to policy
            action = np.random.choice(self.action_space, p=policy_probs)
        else:
            # Greedy action
            action = np.argmax(policy_probs)
        
        action_prob = policy_probs[action]
        
        return action, action_prob
    
    def get_q_values(self, observation: Dict) -> np.ndarray:
        """Get Q-values for all actions given observation."""
        state_key = self._get_state_key(observation)
        
        if self.use_function_approximation:
            with torch.no_grad():
                obs_tensor = self._process_observation(observation)
                q_values = self.q_network(obs_tensor).cpu().numpy().flatten()
        else:
            q_values = self.q_table[state_key]
        
        return q_values
    
    def update(self, batch_data: Dict) -> Dict[str, float]:
        """
        Update Q-values and policy using WoLF-PHC.
        
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
        
        total_q_updates = 0
        total_policy_updates = 0
        winning_states = 0
        losing_states = 0
        
        for i in range(len(observations)):
            state_key = self._get_state_key(observations[i])
            next_state_key = self._get_state_key(next_observations[i])
            action = actions[i]
            reward = rewards[i]
            done = dones[i]
            
            # Update state visit count
            self.state_visits[state_key] += 1
            
            if self.use_function_approximation:
                # Update Q-network
                self._update_q_network(observations[i], action, reward, next_observations[i], done)
                
                # Update policy network
                is_winning = self._is_winning_function_approx(observations[i])
                self._update_policy_network(observations[i], action, is_winning)
            else:
                # Update Q-table
                self._update_q_table(state_key, action, reward, next_state_key, done)
                
                # Update policy using WoLF
                is_winning = self._is_winning_tabular(state_key)
                self._update_policy_tabular(state_key, is_winning)
            
            total_q_updates += 1
            if is_winning:
                winning_states += 1
            else:
                losing_states += 1
            total_policy_updates += 1
        
        return {
            'q_updates': total_q_updates,
            'policy_updates': total_policy_updates,
            'winning_states': winning_states,
            'losing_states': losing_states,
            'win_rate': winning_states / max(1, winning_states + losing_states)
        }
    
    def _update_q_table(self, state_key: str, action: int, reward: float, 
                       next_state_key: str, done: bool):
        """Update Q-table using Q-learning."""
        if not done:
            max_next_q = np.max(self.q_table[next_state_key])
            target = reward + self.gamma * max_next_q
        else:
            target = reward
        
        # Q-learning update
        td_error = target - self.q_table[state_key][action]
        self.q_table[state_key][action] += self.learning_rate * td_error
    
    def _update_q_network(self, observation: Dict, action: int, reward: float,
                         next_observation: Dict, done: bool):
        """Update Q-network using function approximation."""
        obs_tensor = self._process_observation(observation)
        next_obs_tensor = self._process_observation(next_observation)
        
        # Current Q-value
        current_q = self.q_network(obs_tensor)[0, action]
        
        # Target Q-value
        with torch.no_grad():
            if not done:
                next_q_values = self.q_network(next_obs_tensor)
                max_next_q = torch.max(next_q_values)
                target = reward + self.gamma * max_next_q
            else:
                target = torch.tensor(reward).to(self.device)
        
        # Q-learning loss
        loss = F.mse_loss(current_q, target)
        
        # Update Q-network (simplified - would normally use optimizer)
        loss.backward()
        with torch.no_grad():
            for param in self.q_network.parameters():
                if param.grad is not None:
                    param -= self.learning_rate * param.grad
                    param.grad.zero_()
    
    def _is_winning_tabular(self, state_key: str) -> bool:
        """Check if the agent is winning in this state (tabular)."""
        current_policy = self.policy[state_key]
        average_policy = self.average_policy[state_key]
        q_values = self.q_table[state_key]
        
        # Expected value under current policy
        current_value = np.sum(current_policy * q_values)
        
        # Expected value under average policy
        average_value = np.sum(average_policy * q_values)
        
        return current_value > average_value
    
    def _is_winning_function_approx(self, observation: Dict) -> bool:
        """Check if the agent is winning (function approximation)."""
        with torch.no_grad():
            obs_tensor = self._process_observation(observation)
            
            # Current policy and Q-values
            policy_logits = self.policy_network(obs_tensor)
            current_policy = F.softmax(policy_logits, dim=-1).cpu().numpy().flatten()
            q_values = self.q_network(obs_tensor).cpu().numpy().flatten()
            
            # Simple heuristic: compare to uniform policy
            uniform_policy = np.ones(self.action_space) / self.action_space
            
            current_value = np.sum(current_policy * q_values)
            uniform_value = np.sum(uniform_policy * q_values)
            
            return current_value > uniform_value
    
    def _update_policy_tabular(self, state_key: str, is_winning: bool):
        """Update policy using WoLF principle (tabular)."""
        # Choose learning rate based on winning/losing
        delta = self.delta_win if is_winning else self.delta_lose
        
        # Find best action
        q_values = self.q_table[state_key]
        best_action = np.argmax(q_values)
        
        # Update policy towards best action
        policy_change = np.zeros(self.action_space)
        
        # Increase probability of best action
        increase = delta * min(1.0 - self.policy[state_key][best_action], 
                              sum(self.policy[state_key]) - self.policy[state_key][best_action])
        policy_change[best_action] = increase
        
        # Decrease probability of other actions proportionally
        if increase > 0:
            for a in range(self.action_space):
                if a != best_action and self.policy[state_key][a] > 0:
                    decrease = increase * (self.policy[state_key][a] / 
                                         sum(self.policy[state_key][j] for j in range(self.action_space) if j != best_action))
                    policy_change[a] = -decrease
        
        # Apply changes and ensure valid probability distribution
        self.policy[state_key] += policy_change
        self.policy[state_key] = np.clip(self.policy[state_key], 0.0, 1.0)
        self.policy[state_key] /= np.sum(self.policy[state_key])
        
        # Update average policy
        visits = self.state_visits[state_key]
        if visits > 1:
            weight = 1.0 / visits
            self.average_policy[state_key] = (1 - weight) * self.average_policy[state_key] + weight * self.policy[state_key]
    
    def _update_policy_network(self, observation: Dict, action: int, is_winning: bool):
        """Update policy network using WoLF principle."""
        delta = self.delta_win if is_winning else self.delta_lose
        
        obs_tensor = self._process_observation(observation)
        
        # Get current policy
        policy_logits = self.policy_network(obs_tensor)
        current_policy = F.softmax(policy_logits, dim=-1)
        
        # Get Q-values to determine best action
        with torch.no_grad():
            q_values = self.q_network(obs_tensor)
            best_action = torch.argmax(q_values, dim=-1).item()
        
        # Create target policy (move towards best action)
        target_policy = current_policy.clone()
        target_policy[0, best_action] += delta
        target_policy = F.softmax(target_policy, dim=-1)
        
        # Policy gradient loss
        loss = F.kl_div(F.log_softmax(policy_logits, dim=-1), target_policy, reduction='batchmean')
        
        # Update policy network
        loss.backward()
        with torch.no_grad():
            for param in self.policy_network.parameters():
                if param.grad is not None:
                    param -= delta * param.grad
                    param.grad.zero_()
    
    def _get_state_key(self, observation: Dict) -> str:
        """Create a hashable key for the state."""
        if isinstance(observation, dict):
            state_parts = []
            for key, value in sorted(observation.items()):
                if isinstance(value, np.ndarray):
                    # Discretize continuous observations
                    discretized = np.round(value, decimals=2)
                    state_parts.append(f"{key}:{discretized.flatten().tolist()}")
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
            'q_table': dict(self.q_table),
            'policy': dict(self.policy),
            'average_policy': dict(self.average_policy),
            'state_visits': dict(self.state_visits),
            'agent_id': self.agent_id,
            'config': self.config
        }
        
        if self.use_function_approximation:
            checkpoint['q_network_state_dict'] = self.q_network.state_dict()
            checkpoint['policy_network_state_dict'] = self.policy_network.state_dict()
        
        torch.save(checkpoint, f"{path}_wolfphc_agent_{self.agent_id}.pth")
        print(f"Saved WoLF-PHC Agent {self.agent_id} model to {path}_wolfphc_agent_{self.agent_id}.pth")
    
    def load_model(self, path: str):
        """Load the agent's model."""
        checkpoint = torch.load(f"{path}_wolfphc_agent_{self.agent_id}.pth", map_location=self.device)
        
        self.q_table = defaultdict(lambda: np.zeros(self.action_space), checkpoint['q_table'])
        self.policy = defaultdict(lambda: np.ones(self.action_space) / self.action_space, checkpoint['policy'])
        self.average_policy = defaultdict(lambda: np.ones(self.action_space) / self.action_space, checkpoint['average_policy'])
        self.state_visits = defaultdict(int, checkpoint['state_visits'])
        
        if self.use_function_approximation:
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.policy_network.load_state_dict(checkpoint['policy_network_state_dict'])
        
        print(f"Loaded WoLF-PHC Agent {self.agent_id} model from {path}_wolfphc_agent_{self.agent_id}.pth")


class WoLFPHCQNetwork(nn.Module):
    """Q-network for WoLF-PHC with function approximation."""
    
    def __init__(self, obs_space: Dict, action_space: int, config: Dict):
        super().__init__()
        
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


class WoLFPHCPolicyNetwork(nn.Module):
    """Policy network for WoLF-PHC with function approximation."""
    
    def __init__(self, obs_space: Dict, action_space: int, config: Dict):
        super().__init__()
        
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
        """Forward pass through policy network."""
        return self.network(obs)


class WoLFPHCReplayBuffer:
    """Simple replay buffer for WoLF-PHC."""
    
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
            'dones': [t['done'] for t in batch]
        }
        
        return batch_data
    
    def __len__(self):
        return len(self.buffer)


class WoLFPHC(MARLAlgorithm):
    """
    WoLF-PHC (Win or Learn Fast - Policy Hill Climbing) algorithm.
    
    Uses adaptive learning rates - faster learning when losing,
    slower when winning to promote convergence.
    """
    
    def __init__(self, env, config: Dict, device: torch.device):
        """
        Initialize WoLF-PHC algorithm.
        
        Args:
            env: Multi-agent environment
            config: Configuration dictionary
            device: PyTorch device
        """
        super().__init__(env, config, device)
        
        # WoLF-PHC specific parameters
        self.batch_size = config.get('batch_size', 32)
        self.memory_size = config.get('memory_size', 10000)
        self.train_start = config.get('train_start', 1000)
        self.update_interval = config.get('update_interval', 1)
        
        # Create replay buffers for each agent
        self.replay_buffers = [
            WoLFPHCReplayBuffer(self.memory_size) for _ in range(self.n_agents)
        ]
        
        print(f"Initialized WoLF-PHC")
        print(f"Batch size: {self.batch_size}")
        print(f"Memory size: {self.memory_size}")
    
    def _create_agents(self):
        """Create WoLF-PHC agents."""
        obs_space = self._get_obs_space()
        action_space = len(self.env.actions)
        
        for i in range(self.n_agents):
            agent = WoLFPHCAgent(i, obs_space, action_space, self.config)
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
        
        return {
            'episode_length': step_count,
            'total_reward': total_reward
        }
    
    def train_step(self, rollout_data: Dict) -> Dict[str, float]:
        """Perform WoLF-PHC training step."""
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
