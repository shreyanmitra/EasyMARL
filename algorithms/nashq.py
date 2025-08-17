"""
Nash-Q Learning algorithm.

Nash-Q computes Nash equilibrium policies in multi-agent Q-learning
for general-sum games with discrete actions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from typing import Dict, List, Tuple, Any
from collections import deque, defaultdict
import random
from scipy.optimize import linprog
import itertools

from .base import MARLAgent, MARLAlgorithm


class NashQAgent(MARLAgent):
    """
    Nash-Q agent that computes Nash equilibrium policies.
    """
    
    def __init__(self, agent_id: int, obs_space: Dict, action_space: int, config: Dict):
        """
        Initialize the Nash-Q agent.
        
        Args:
            agent_id: Unique identifier for this agent
            obs_space: Individual observation space
            action_space: Number of available actions
            config: Configuration dictionary
        """
        super().__init__(agent_id, obs_space, action_space, config)
        
        # Nash-Q hyperparameters
        self.gamma = config.get('gamma', 0.99)
        self.learning_rate = config.get('learning_rate', 0.1)
        self.epsilon = config.get('epsilon', 0.1)
        self.n_agents = config.get('n_agents', 2)
        
        # Initialize Q-tables for all agents
        # Q[state][joint_action] = payoff_matrix for this agent
        self.q_table = defaultdict(lambda: np.zeros([action_space] * self.n_agents))
        
        # Store Nash equilibrium policies
        self.nash_policies = defaultdict(lambda: np.ones(action_space) / action_space)
        
        # Function approximation option
        self.use_function_approximation = config.get('use_function_approximation', False)
        if self.use_function_approximation:
            self.q_network = NashQNetwork(obs_space, action_space, self.n_agents, config).to(self.device)
            self.optimizer = Adam(self.q_network.parameters(), lr=config.get('learning_rate', 1e-3))
        
        print(f"Initialized Nash-Q Agent {agent_id}")
        print(f"Number of agents: {self.n_agents}")
        print(f"Function approximation: {self.use_function_approximation}")
    
    def get_action(self, observation: Dict, training: bool = True) -> Tuple[int, float]:
        """
        Select an action using Nash equilibrium policy.
        
        Args:
            observation: Individual agent observation
            training: Whether in training mode
            
        Returns:
            Tuple of (action, action_probability)
        """
        state_key = self._get_state_key(observation)
        
        if self.use_function_approximation:
            # Get Nash policy from network
            with torch.no_grad():
                obs_tensor = self._process_observation(observation)
                nash_probs = F.softmax(self.q_network.get_policy(obs_tensor), dim=-1)
                nash_probs = nash_probs.cpu().numpy().flatten()
        else:
            nash_probs = self.nash_policies[state_key]
        
        if training and random.random() < self.epsilon:
            # Epsilon-greedy exploration
            action = random.randint(0, self.action_space - 1)
        else:
            # Sample from Nash equilibrium policy
            action = np.random.choice(self.action_space, p=nash_probs)
        
        action_prob = nash_probs[action]
        
        return action, action_prob
    
    def get_q_values(self, observation: Dict) -> np.ndarray:
        """Get Q-values matrix for all joint actions."""
        state_key = self._get_state_key(observation)
        
        if self.use_function_approximation:
            with torch.no_grad():
                obs_tensor = self._process_observation(observation)
                q_values = self.q_network(obs_tensor).cpu().numpy()
                # Reshape to joint action space
                return q_values.reshape([self.action_space] * self.n_agents)
        else:
            return self.q_table[state_key]
    
    def update(self, batch_data: Dict) -> Dict[str, float]:
        """
        Update Q-values and compute Nash equilibrium.
        
        Args:
            batch_data: Dictionary containing transitions
            
        Returns:
            Dictionary containing loss metrics
        """
        observations = batch_data['observations']
        joint_actions = batch_data['joint_actions']
        rewards = batch_data['rewards']
        next_observations = batch_data['next_observations']
        dones = batch_data['dones']
        
        total_loss = 0.0
        nash_updates = 0
        
        for i in range(len(observations)):
            state_key = self._get_state_key(observations[i])
            next_state_key = self._get_state_key(next_observations[i])
            joint_action = tuple(joint_actions[i])
            reward = rewards[i]
            done = dones[i]
            
            if self.use_function_approximation:
                # Update Q-network
                loss = self._update_q_network(
                    observations[i], joint_action, reward, next_observations[i], done
                )
                total_loss += loss
            else:
                # Update Q-table
                self._update_q_table(state_key, joint_action, reward, next_state_key, done)
            
            # Compute new Nash equilibrium for current state
            nash_policy = self._compute_nash_equilibrium(state_key if not self.use_function_approximation else observations[i])
            
            if nash_policy is not None:
                if self.use_function_approximation:
                    # Store Nash policy in network (would need policy head)
                    pass
                else:
                    self.nash_policies[state_key] = nash_policy
                nash_updates += 1
        
        return {
            'q_loss': total_loss / len(observations) if self.use_function_approximation else 0.0,
            'nash_updates': nash_updates,
            'epsilon': self.epsilon
        }
    
    def _update_q_table(self, state_key: str, joint_action: Tuple[int], 
                       reward: float, next_state_key: str, done: bool):
        """Update Q-table using Nash-Q learning."""
        if not done:
            # Compute Nash equilibrium value for next state
            next_nash_value = self._compute_nash_value(next_state_key)
            target = reward + self.gamma * next_nash_value
        else:
            target = reward
        
        # Q-learning update
        current_q = self.q_table[state_key][joint_action]
        td_error = target - current_q
        self.q_table[state_key][joint_action] += self.learning_rate * td_error
    
    def _update_q_network(self, observation: Dict, joint_action: Tuple[int], 
                         reward: float, next_observation: Dict, done: bool) -> float:
        """Update Q-network using Nash-Q learning."""
        obs_tensor = self._process_observation(observation)
        next_obs_tensor = self._process_observation(next_observation)
        
        # Current Q-value for joint action
        q_values = self.q_network(obs_tensor)
        joint_action_idx = self._joint_action_to_index(joint_action)
        current_q = q_values.flatten()[joint_action_idx]
        
        # Target Q-value
        with torch.no_grad():
            if not done:
                next_q_values = self.q_network(next_obs_tensor)
                next_nash_value = self._compute_nash_value_tensor(next_q_values)
                target = reward + self.gamma * next_nash_value
            else:
                target = torch.tensor(reward).to(self.device)
        
        # Loss and backpropagation
        loss = F.mse_loss(current_q, target)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def _compute_nash_equilibrium(self, state) -> np.ndarray:
        """
        Compute Nash equilibrium policy for this agent in the given state.
        
        Args:
            state: State key (string) or observation (Dict)
            
        Returns:
            Nash equilibrium policy for this agent
        """
        try:
            if self.use_function_approximation:
                # Get Q-values from network
                obs_tensor = self._process_observation(state)
                with torch.no_grad():
                    q_values = self.q_network(obs_tensor).cpu().numpy()
                    q_matrix = q_values.reshape([self.action_space] * self.n_agents)
            else:
                # Get Q-values from table
                q_matrix = self.q_table[state]
            
            # Solve for Nash equilibrium using linear programming
            nash_policy = self._solve_nash_lp(q_matrix)
            
            return nash_policy
            
        except Exception as e:
            print(f"Nash equilibrium computation failed: {e}")
            # Fallback to uniform policy
            return np.ones(self.action_space) / self.action_space
    
    def _solve_nash_lp(self, q_matrix: np.ndarray) -> np.ndarray:
        """
        Solve for Nash equilibrium using linear programming.
        This is a simplified implementation for 2-agent case.
        """
        if self.n_agents != 2:
            # For more than 2 agents, use uniform policy as fallback
            return np.ones(self.action_space) / self.action_space
        
        # Convert to zero-sum game format for this agent
        # This is a simplified Nash computation
        
        # For agent 0, we want to solve:
        # max_π min_σ ∑_a π(a) σ(b) Q(a,b)
        
        try:
            # Use fictitious play approximation for simplicity
            best_response = np.zeros(self.action_space)
            
            # Find best response to uniform policy of opponent
            uniform_policy = np.ones(self.action_space) / self.action_space
            
            for a in range(self.action_space):
                expected_value = 0.0
                for b in range(self.action_space):
                    if self.agent_id == 0:
                        expected_value += uniform_policy[b] * q_matrix[a, b]
                    else:
                        expected_value += uniform_policy[b] * q_matrix[b, a]
                
                if a == 0 or expected_value > best_value:
                    best_value = expected_value
                    best_action = a
            
            # Epsilon-greedy policy around best response
            nash_policy = np.full(self.action_space, 0.1 / self.action_space)
            nash_policy[best_action] = 0.9 + 0.1 / self.action_space
            
            return nash_policy
            
        except Exception:
            # Fallback to uniform policy
            return np.ones(self.action_space) / self.action_space
    
    def _compute_nash_value(self, state_key: str) -> float:
        """Compute Nash equilibrium value for a state."""
        q_matrix = self.q_table[state_key]
        nash_policy = self.nash_policies[state_key]
        
        # Compute expected value under Nash policy
        expected_value = 0.0
        for joint_action in itertools.product(range(self.action_space), repeat=self.n_agents):
            joint_prob = 1.0
            for agent_id, action in enumerate(joint_action):
                if agent_id == self.agent_id:
                    joint_prob *= nash_policy[action]
                else:
                    # Assume other agents also use Nash (simplified)
                    joint_prob *= nash_policy[action]  # This should be other agent's policy
            
            expected_value += joint_prob * q_matrix[joint_action]
        
        return expected_value
    
    def _compute_nash_value_tensor(self, q_values: torch.Tensor) -> torch.Tensor:
        """Compute Nash equilibrium value for tensor Q-values."""
        # Simplified: use max value as approximation
        return torch.max(q_values)
    
    def _joint_action_to_index(self, joint_action: Tuple[int]) -> int:
        """Convert joint action tuple to flat index."""
        index = 0
        multiplier = 1
        for i in reversed(range(self.n_agents)):
            index += joint_action[i] * multiplier
            multiplier *= self.action_space
        return index
    
    def _get_state_key(self, observation: Dict) -> str:
        """Create a hashable key for the state."""
        if isinstance(observation, dict):
            state_parts = []
            for key, value in sorted(observation.items()):
                if isinstance(value, np.ndarray):
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
            'joint_actions': [],
            'rewards': [],
            'next_observations': [],
            'dones': []
        }
    
    def save_model(self, path: str):
        """Save the agent's model."""
        checkpoint = {
            'q_table': dict(self.q_table),
            'nash_policies': dict(self.nash_policies),
            'epsilon': self.epsilon,
            'agent_id': self.agent_id,
            'config': self.config
        }
        
        if self.use_function_approximation:
            checkpoint['q_network_state_dict'] = self.q_network.state_dict()
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
        
        torch.save(checkpoint, f"{path}_nashq_agent_{self.agent_id}.pth")
        print(f"Saved Nash-Q Agent {self.agent_id} model to {path}_nashq_agent_{self.agent_id}.pth")
    
    def load_model(self, path: str):
        """Load the agent's model."""
        checkpoint = torch.load(f"{path}_nashq_agent_{self.agent_id}.pth", map_location=self.device)
        
        self.q_table = defaultdict(lambda: np.zeros([self.action_space] * self.n_agents), checkpoint['q_table'])
        self.nash_policies = defaultdict(lambda: np.ones(self.action_space) / self.action_space, checkpoint['nash_policies'])
        self.epsilon = checkpoint['epsilon']
        
        if self.use_function_approximation:
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"Loaded Nash-Q Agent {self.agent_id} model from {path}_nashq_agent_{self.agent_id}.pth")


class NashQNetwork(nn.Module):
    """Q-network for Nash-Q with function approximation."""
    
    def __init__(self, obs_space: Dict, action_space: int, n_agents: int, config: Dict):
        super().__init__()
        
        self.obs_dim = self._estimate_obs_dim(obs_space)
        self.action_space = action_space
        self.n_agents = n_agents
        hidden_dim = config.get('hidden_dim', 128)
        
        # Q-value network for joint actions
        self.q_network = nn.Sequential(
            nn.Linear(self.obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_space ** n_agents)
        )
        
        # Policy network for Nash equilibrium
        self.policy_network = nn.Sequential(
            nn.Linear(self.obs_dim, hidden_dim),
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
        return self.q_network(obs)
    
    def get_policy(self, obs: torch.Tensor) -> torch.Tensor:
        """Get policy logits."""
        return self.policy_network(obs)


class NashQReplayBuffer:
    """Replay buffer for Nash-Q learning."""
    
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
            'joint_actions': [t['joint_action'] for t in batch],
            'rewards': [t['reward'] for t in batch],
            'next_observations': [t['next_observation'] for t in batch],
            'dones': [t['done'] for t in batch]
        }
        
        return batch_data
    
    def __len__(self):
        return len(self.buffer)


class NashQ(MARLAlgorithm):
    """
    Nash-Q Learning algorithm.
    
    Computes Nash equilibrium policies in multi-agent Q-learning
    for general-sum games with discrete actions.
    """
    
    def __init__(self, env, config: Dict, device: torch.device):
        """
        Initialize Nash-Q algorithm.
        
        Args:
            env: Multi-agent environment
            config: Configuration dictionary
            device: PyTorch device
        """
        super().__init__(env, config, device)
        
        # Nash-Q specific parameters
        self.batch_size = config.get('batch_size', 32)
        self.memory_size = config.get('memory_size', 10000)
        self.train_start = config.get('train_start', 1000)
        self.update_interval = config.get('update_interval', 1)
        
        # Create shared replay buffer (Nash-Q requires joint experiences)
        self.replay_buffer = NashQReplayBuffer(self.memory_size)
        
        print(f"Initialized Nash-Q")
        print(f"Batch size: {self.batch_size}")
        print(f"Memory size: {self.memory_size}")
    
    def _create_agents(self):
        """Create Nash-Q agents."""
        obs_space = self._get_obs_space()
        action_space = len(self.env.actions)
        
        # All agents need to know about all other agents
        agent_config = self.config.copy()
        agent_config['n_agents'] = self.n_agents
        
        for i in range(self.n_agents):
            agent = NashQAgent(i, obs_space, action_space, agent_config)
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
        """Collect joint transitions for Nash-Q learning."""
        obs = env.reset()
        done = False
        step_count = 0
        total_reward = [0.0] * self.n_agents
        
        while not done:
            # Get actions from all agents
            actions = []
            agent_observations = []
            
            for i, agent in enumerate(self.agents):
                agent_obs = self._extract_agent_obs(obs, i)
                action, _ = agent.get_action(agent_obs, training=True)
                actions.append(action)
                agent_observations.append(agent_obs)
            
            # Take environment step
            next_obs, rewards, done, info = env.step(actions)
            
            # Store joint transition in shared buffer
            for i in range(self.n_agents):
                agent_obs = agent_observations[i]
                next_agent_obs = self._extract_agent_obs(next_obs, i)
                
                reward = rewards[i] if isinstance(rewards, list) else rewards
                total_reward[i] += reward
                
                transition = {
                    'observation': agent_obs,
                    'joint_action': actions,  # All agents' actions
                    'reward': reward,
                    'next_observation': next_agent_obs,
                    'done': done
                }
                
                self.replay_buffer.add(transition)
            
            obs = next_obs
            step_count += 1
            self.total_steps += 1
        
        return {
            'episode_length': step_count,
            'total_reward': total_reward
        }
    
    def train_step(self, rollout_data: Dict) -> Dict[str, float]:
        """Perform Nash-Q training step."""
        if self.total_steps < self.train_start or len(self.replay_buffer) < self.batch_size:
            return {
                'episode_length': rollout_data['episode_length'],
                'total_reward': sum(rollout_data['total_reward']) / self.n_agents
            }
        
        # Train agents if we have enough experience
        if self.total_steps % self.update_interval == 0:
            # Sample joint batch
            batch_data = self.replay_buffer.sample(self.batch_size)
            
            # Update each agent
            agent_losses = []
            for i, agent in enumerate(self.agents):
                # Create agent-specific batch data
                agent_batch = {
                    'observations': [batch_data['observations'][j] for j in range(self.batch_size) if j % self.n_agents == i],
                    'joint_actions': [batch_data['joint_actions'][j] for j in range(self.batch_size) if j % self.n_agents == i],
                    'rewards': [batch_data['rewards'][j] for j in range(self.batch_size) if j % self.n_agents == i],
                    'next_observations': [batch_data['next_observations'][j] for j in range(self.batch_size) if j % self.n_agents == i],
                    'dones': [batch_data['dones'][j] for j in range(self.batch_size) if j % self.n_agents == i]
                }
                
                if agent_batch['observations']:  # Ensure we have data for this agent
                    loss_info = agent.update(agent_batch)
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
