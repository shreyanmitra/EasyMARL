"""
(C) Shreyan Mitra, based on starter code by Natasha Jaques

Value Decomposition Networks (VDN) Algorithm

VDN is the foundational algorithm for value-based multi-agent reinforcement learning.
It's conceptually simple but powerful, making it an excellent starting point for
understanding how individual agents can learn to cooperate without explicit communication.

Key Innovation - Additive Value Decomposition:
VDN decomposes the joint team Q-value into individual agent Q-values through simple addition:
Q_total(s, u) = Q₁(s₁, u₁) + Q₂(s₂, u₂) + ... + Qₙ(sₙ, uₙ)

How VDN Works:
1. Each agent learns its own Q-function based on local observations
2. The team's total Q-value is the sum of individual Q-values
3. Joint actions are selected by each agent taking their individually best action
4. Training uses global rewards but maintains decentralized execution

When to Use VDN:
✅ Cooperative tasks with shared rewards
✅ When you want simple, interpretable cooperation
✅ Discrete action spaces
✅ When communication during execution is not possible
✅ As a baseline for more complex methods

Key Advantages:
✅ Simple and easy to understand
✅ Computationally efficient
✅ Naturally satisfies Individual-Global-Max (IGM) principle
✅ Scales well to many agents
✅ Strong theoretical foundations

Limitations:
❌ Limited representational capacity (only additive interactions)
❌ Cannot capture complex agent interactions
❌ May struggle with coordination requiring timing or sequencing

Comparison with Other Algorithms:
- vs QMIX: Simpler but less expressive (QMIX uses non-linear mixing)
- vs Independent Q-Learning: Better credit assignment through shared rewards
- vs MAPPO: Value-based vs policy gradient approach

For MARL Beginners:
VDN is perfect for learning MARL concepts! It extends single-agent Q-learning
to multi-agent settings in the most straightforward way possible. Master this
before moving to more complex algorithms like QMIX or MAPPO.

Paper: "Value-Decomposition Networks For Cooperative Multi-Agent Learning" (2017)
Use Cases: Simple coordination tasks, resource allocation, basic teamwork scenarios
"""

# Import necessary libraries for deep learning and multi-agent systems
import torch                    # PyTorch for neural networks
import torch.nn as nn           # Neural network modules
import torch.nn.functional as F # Activation functions and utilities
from torch.optim import Adam    # Adam optimizer for gradient-based learning
import numpy as np              # Numerical computations
from typing import Dict, List, Tuple, Any  # Type hints for code clarity
import copy                     # For creating deep copies of networks

# Import base classes from our MARL framework
from .base import MARLAgent, MARLAlgorithm


class VDNAgent(MARLAgent):
    """
    Value Decomposition Networks Agent.
    
    This agent learns its own Q-function that estimates the value of taking
    different actions from local observations. The key insight is that when
    all agents do this and the team reward is shared, they naturally learn
    to cooperate through the credit assignment of shared rewards.
    
    Key Components:
    1. Individual Q-Network: Q(local_observation, action) → value estimate
    2. Target Network: Stable target for temporal difference learning
    3. Epsilon-Greedy Exploration: Balance between exploitation and exploration
    4. Experience Replay: Learn from past experiences
    
    Architecture Insight:
    - Q-Network: Input = local observation → Output = Q-values for each action
    - Target Network: Delayed copy of Q-network for stable learning
    - Additive Combination: Team value = sum of individual Q-values
    
    For MARL Beginners:
    Think of this as a single-agent Q-learning agent that happens to be on a team.
    It learns the value of its actions, but because all agents share the same reward,
    they naturally learn to work together to maximize the team's performance.
    """
    
    def __init__(self, agent_id: int, obs_space: Dict, action_space: int, config: Dict):
        """
        Initialize the VDN agent with Q-networks for value-based learning.
        
        Args:
            agent_id (int): Unique identifier for this agent
            obs_space (Dict): Local observation space for this agent
                             Example: {'image': (7, 7, 3), 'direction': 4}
            action_space (int): Number of actions this agent can take
                               Example: 6 for MultiGrid environments
            config (Dict): Configuration containing hyperparameters
                          Example: {'epsilon_start': 1.0, 'epsilon_decay': 0.995, ...}
        
        For Beginners:
        This creates an agent that learns the value of different actions
        from its local perspective, similar to single-agent Q-learning.
        """
        # Call parent class constructor to set up basic agent properties
        super().__init__(agent_id, obs_space, action_space, config)
        
        # Epsilon-Greedy Exploration Parameters
        # These control the balance between exploration (trying new actions) and
        # exploitation (using known good actions)
        self.epsilon = config.get('epsilon_start', 1.0)        # Start with high exploration
        self.epsilon_end = config.get('epsilon_end', 0.05)     # End with low exploration
        self.epsilon_decay = config.get('epsilon_decay', 0.995) # How fast to reduce exploration
        
        # Q-Network Architecture
        # Main network: learns to estimate Q-values for state-action pairs
        self.q_network = VDNQNetwork(obs_space, action_space, config).to(self.device)
        
        # Target network: provides stable targets for temporal difference learning
        # This is a key technique in deep Q-learning to prevent instability
        self.target_q_network = copy.deepcopy(self.q_network)
        
        # Debug information
        print(f"Initialized VDN Agent {agent_id} with {sum(p.numel() for p in self.q_network.parameters())} parameters")
    
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
    
    def get_target_q_values(self, observation: Dict) -> torch.Tensor:
        """Get target Q-values for the given observation."""
        obs_tensor = self._process_observation(observation)
        return self.target_q_network(obs_tensor)
    
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
    
    def update_epsilon(self):
        """Update epsilon for exploration."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def update_target_network(self):
        """Hard update of target network."""
        self.target_q_network.load_state_dict(self.q_network.state_dict())
    
    def reset_memory(self):
        """VDN uses centralized replay buffer."""
        pass
    
    def save_model(self, path: str):
        """Save the agent's model to disk."""
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_q_network_state_dict': self.target_q_network.state_dict(),
            'epsilon': self.epsilon,
            'agent_id': self.agent_id,
            'config': self.config
        }
        torch.save(checkpoint, f"{path}_vdn_agent_{self.agent_id}.pth")
        print(f"Saved VDN Agent {self.agent_id} model to {path}_vdn_agent_{self.agent_id}.pth")
    
    def load_model(self, path: str):
        """Load the agent's model from disk."""
        checkpoint = torch.load(f"{path}_vdn_agent_{self.agent_id}.pth", map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_q_network.load_state_dict(checkpoint['target_q_network_state_dict'])
        self.epsilon = checkpoint['epsilon']
        print(f"Loaded VDN Agent {self.agent_id} model from {path}_vdn_agent_{self.agent_id}.pth")


class VDNQNetwork(nn.Module):
    """Q-network for individual agents in VDN."""
    
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


class VDN(MARLAlgorithm):
    """
    Value Decomposition Networks (VDN) algorithm.
    
    VDN learns individual Q-values and sums them to get joint Q-values,
    satisfying the IGM principle through additive decomposition.
    """
    
    def __init__(self, env, config: Dict, device: torch.device):
        """
        Initialize the VDN algorithm.
        
        Args:
            env: Multi-agent environment
            config: Configuration dictionary
            device: PyTorch device
        """
        super().__init__(env, config, device)
        
        # VDN specific parameters
        self.batch_size = config.get('batch_size', 32)
        self.buffer_size = config.get('buffer_size', 50000)
        self.learning_starts = config.get('learning_starts', 1000)
        self.train_freq = config.get('train_freq', 4)
        self.target_update_freq = config.get('target_update_freq', 200)
        self.gamma = config.get('gamma', 0.99)
        self.lr = config.get('lr', 5e-4)
        self.grad_norm_clip = config.get('grad_norm_clip', 10.0)
        
        # Create centralized replay buffer
        obs_shape = self._get_obs_shape()
        self.replay_buffer = VDNEpisodeReplayBuffer(
            self.buffer_size, obs_shape, self.n_agents, config.get('max_episode_length', 100)
        )
        
        # Optimizer for all networks
        params = []
        for agent in self.agents:
            params.extend(list(agent.q_network.parameters()))
        
        self.optimizer = Adam(params, lr=self.lr)
        
        self.steps_done = 0
        
        print(f"Initialized VDN")
        print(f"Value decomposition: Additive (Q_tot = sum(Q_i))")
    
    def _create_agents(self):
        """Create VDN agents."""
        obs_space = self._get_obs_space()
        action_space = len(self.env.actions)
        
        for i in range(self.n_agents):
            agent = VDNAgent(i, obs_space, action_space, self.config)
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
    
    def _get_obs_shape(self) -> Tuple:
        """Get observation shape."""
        obs_space = self._get_obs_space()
        total_dim = 0
        for key, value in obs_space.items():
            if hasattr(value, 'shape'):
                total_dim += np.prod(value.shape)
            elif isinstance(value, np.ndarray):
                total_dim += np.prod(value.shape)
            else:
                total_dim += 1
        return (total_dim,)
    
    def collect_rollout(self, env) -> Dict:
        """Collect a full episode for VDN training."""
        episode_obs = []
        episode_actions = []
        episode_rewards = []
        episode_dones = []
        
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        while not done:
            # Store current state
            episode_obs.append(obs)
            
            # Get actions from all agents
            actions = []
            for i, agent in enumerate(self.agents):
                # Extract agent observation
                agent_obs = self._extract_agent_obs(obs, i)
                action = agent.get_action(agent_obs, training=True)
                actions.append(action)
            
            episode_actions.append(actions)
            
            # Take environment step
            next_obs, rewards, done, info = env.step(actions)
            
            # Store rewards and done
            if isinstance(rewards, list):
                total_reward = sum(rewards)
                episode_rewards.append(rewards)
            else:
                total_reward = rewards * self.n_agents  # Assume shared reward
                episode_rewards.append([rewards] * self.n_agents)
            
            episode_dones.append([done] * self.n_agents)
            
            episode_reward += total_reward
            episode_length += 1
            obs = next_obs
            self.steps_done += 1
        
        # Store episode in replay buffer
        self.replay_buffer.add_episode({
            'observations': episode_obs,
            'actions': episode_actions,
            'rewards': episode_rewards,
            'dones': episode_dones
        })
        
        # Update exploration
        for agent in self.agents:
            agent.update_epsilon()
        
        return {
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'epsilon': self.agents[0].epsilon
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
        """Perform VDN training step."""
        metrics = {
            'episode_reward': rollout_data['episode_reward'],
            'episode_length': rollout_data['episode_length'],
            'epsilon': rollout_data['epsilon'],
            'buffer_size': len(self.replay_buffer)
        }
        
        # Only train if we have enough samples
        if len(self.replay_buffer) < self.learning_starts:
            return metrics
        
        # Train every few steps
        if self.steps_done % self.train_freq == 0:
            loss = self._update_networks()
            metrics['loss'] = loss
        
        # Update target networks
        if self.steps_done % self.target_update_freq == 0:
            self._update_target_networks()
        
        return metrics
    
    def _update_networks(self) -> float:
        """Update Q-networks using VDN."""
        # Sample batch of episodes
        batch = self.replay_buffer.sample(self.batch_size)
        
        # Move to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(self.device)
        
        # Compute current Q-values for each agent
        current_q_values = []
        for i, agent in enumerate(self.agents):
            agent_q_vals = []
            for t in range(batch['episode_length']):
                obs_t = batch['observations'][t][:, i]  # [batch_size, obs_dim]
                q_vals = agent.q_network(obs_t)  # [batch_size, n_actions]
                actions_t = batch['actions'][t][:, i].long()  # [batch_size]
                q_taken = q_vals.gather(1, actions_t.unsqueeze(-1)).squeeze(-1)  # [batch_size]
                agent_q_vals.append(q_taken)
            
            current_q_values.append(torch.stack(agent_q_vals))  # [episode_length, batch_size]
        
        current_q_values = torch.stack(current_q_values)  # [n_agents, episode_length, batch_size]
        
        # Compute target Q-values for each agent
        target_q_values = []
        for i, agent in enumerate(self.agents):
            agent_target_q_vals = []
            for t in range(batch['episode_length']):
                if t == batch['episode_length'] - 1:
                    # Last step - use reward only
                    target_q_vals = batch['rewards'][t][:, i]
                else:
                    # Next step target
                    next_obs_t = batch['observations'][t + 1][:, i]
                    next_q_vals = agent.target_q_network(next_obs_t)
                    max_next_q = next_q_vals.max(dim=1)[0]
                    target_q_vals = batch['rewards'][t][:, i] + self.gamma * max_next_q * (1 - batch['dones'][t][:, i])
                
                agent_target_q_vals.append(target_q_vals)
            
            target_q_values.append(torch.stack(agent_target_q_vals))
        
        target_q_values = torch.stack(target_q_values)  # [n_agents, episode_length, batch_size]
        
        # VDN: Sum individual Q-values to get joint Q-values
        joint_q_values = current_q_values.sum(dim=0)  # [episode_length, batch_size]
        target_joint_q_values = target_q_values.sum(dim=0)  # [episode_length, batch_size]
        
        # Compute loss
        loss = F.mse_loss(joint_q_values, target_joint_q_values.detach())
        
        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            [p for agent in self.agents for p in agent.q_network.parameters()],
            self.grad_norm_clip
        )
        
        self.optimizer.step()
        
        return loss.item()
    
    def _update_target_networks(self):
        """Update target networks."""
        for agent in self.agents:
            agent.update_target_network()


class VDNEpisodeReplayBuffer:
    """
    Episode-based replay buffer for VDN.
    """
    
    def __init__(self, capacity: int, obs_shape: Tuple, n_agents: int, max_episode_length: int):
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.n_agents = n_agents
        self.max_episode_length = max_episode_length
        
        self.episodes = []
        self.ptr = 0
    
    def add_episode(self, episode: Dict):
        """Add a complete episode to the buffer."""
        # Pad episode to max length if necessary
        episode_length = len(episode['observations'])
        
        if episode_length < self.max_episode_length:
            # Pad with zeros
            for _ in range(self.max_episode_length - episode_length):
                episode['observations'].append(episode['observations'][-1])  # Repeat last obs
                episode['actions'].append([0] * self.n_agents)  # Dummy actions
                episode['rewards'].append([0.0] * self.n_agents)  # Zero rewards
                episode['dones'].append([True] * self.n_agents)  # Mark as done
        
        if len(self.episodes) < self.capacity:
            self.episodes.append(episode)
        else:
            self.episodes[self.ptr] = episode
            self.ptr = (self.ptr + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a batch of episodes."""
        indices = np.random.choice(len(self.episodes), batch_size, replace=False)
        
        batch = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'dones': [],
            'episode_length': self.max_episode_length
        }
        
        for idx in indices:
            episode = self.episodes[idx]
            
            # Convert to tensors and stack
            for t in range(self.max_episode_length):
                if t == 0:
                    # Initialize lists for this timestep
                    batch['observations'].append([])
                    batch['actions'].append([])
                    batch['rewards'].append([])
                    batch['dones'].append([])
                
                # Process observations for all agents at this timestep
                obs_t = []
                for i in range(self.n_agents):
                    agent_obs = self._extract_agent_obs(episode['observations'][t], i)
                    obs_flat = self._flatten_obs(agent_obs)
                    obs_t.append(obs_flat)
                
                batch['observations'][t].append(obs_t)
                batch['actions'][t].append(episode['actions'][t])
                batch['rewards'][t].append(episode['rewards'][t])
                batch['dones'][t].append(episode['dones'][t])
        
        # Convert to tensors
        for t in range(self.max_episode_length):
            batch['observations'][t] = torch.FloatTensor(batch['observations'][t])  # [batch_size, n_agents, obs_dim]
            batch['actions'][t] = torch.LongTensor(batch['actions'][t])  # [batch_size, n_agents]
            batch['rewards'][t] = torch.FloatTensor(batch['rewards'][t])  # [batch_size, n_agents]
            batch['dones'][t] = torch.FloatTensor(batch['dones'][t])  # [batch_size, n_agents]
        
        return batch
    
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
    
    def _flatten_obs(self, obs) -> np.ndarray:
        """Flatten observation to 1D array."""
        if isinstance(obs, dict):
            obs_flat = []
            for key, value in obs.items():
                if isinstance(value, np.ndarray):
                    obs_flat.append(value.flatten())
                else:
                    obs_flat.append(np.array([value]))
            return np.concatenate(obs_flat)
        else:
            return np.array(obs).flatten()
    
    def __len__(self) -> int:
        return len(self.episodes)
