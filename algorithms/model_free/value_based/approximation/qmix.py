"""
(C) Shreyan Mitra, based on starter code by Natasha Jaques

QMIX Algorithm for Multi-Agent Reinforcement Learning

QMIX is one of the most successful and widely-used MARL algorithms, particularly
effective in cooperative multi-agent scenarios. It solves the challenging problem
of credit assignment in multi-agent systems.

Key Innovation - Individual-Global-Max (IGM) Principle:
QMIX ensures that the optimal joint action (what's best for the team) corresponds
to each agent taking their individually optimal action. This means:
- Each agent can act independently using only local observations
- The team's joint action is guaranteed to be optimal
- No communication needed during execution (decentralized execution)

How QMIX Works:
1. Each agent has its own Q-network (estimates action values from local observations)
2. A mixing network combines individual Q-values into a joint team Q-value
3. The mixing network has only positive weights (ensures IGM principle)
4. Training is centralized (uses global information) but execution is decentralized

When to Use QMIX:
✅ Cooperative tasks where agents work toward a common goal
✅ Partial observability (agents can't see everything)
✅ Need for decentralized execution (no communication during play)
✅ Discrete action spaces

Paper: "QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent RL" (2018)
Use Cases: StarCraft II micromanagement, traffic coordination, swarm robotics

For MARL Beginners:
QMIX is an excellent starting point for learning MARL. It's conceptually clear,
theoretically grounded, and works well in practice. Start here before moving
to more complex algorithms like MADDPG or MAPPO.
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
from .base import MARLAgent, MARLAlgorithm, ReplayBuffer


class QMIXAgent(MARLAgent):
    """
    Individual Q-Learning Agent for QMIX Algorithm.
    
    Each agent in QMIX has its own Q-network that learns to estimate action values
    based on its local observations. The key insight is that even though agents
    act independently, their learning is coordinated through the QMIX mixing network.
    
    Key Properties:
    - Local Observations: Only sees partial environment state
    - Individual Q-Network: Learns Q(observation, action) for its own actions
    - Epsilon-Greedy: Balances exploration vs exploitation
    - No Individual Optimizer: Learning coordinated by QMIX algorithm
    
    For MARL Beginners:
    Think of each agent as a team member who learns their individual role
    while the coach (QMIX) ensures everyone works well together.
    """
    
    def __init__(self, agent_id: int, obs_space: Dict, action_space: int, config: Dict):
        """
        Initialize the QMIX agent with its individual Q-network.
        
        Args:
            agent_id (int): Unique identifier for this agent (0, 1, 2, ...)
            obs_space (Dict): Local observation space for this agent
                             Example: {'image': (7, 7, 3), 'direction': 4}
            action_space (int): Number of actions this agent can take
                               Example: 6 for MultiGrid (move up, down, left, right, toggle, done)
            config (Dict): Configuration containing hyperparameters
                          Example: {'epsilon_start': 1.0, 'epsilon_decay': 0.995, ...}
        
        For Beginners:
        This sets up one team member with their own "skill set" (Q-network)
        and exploration strategy (epsilon-greedy).
        """
        # Call parent class constructor to set up basic agent properties
        super().__init__(agent_id, obs_space, action_space, config)
        
        # Exploration Parameters for Epsilon-Greedy Action Selection
        # Start with high exploration and gradually reduce it
        self.epsilon = config.get('epsilon_start', 1.0)        # Current exploration rate (100% initially)
        self.epsilon_end = config.get('epsilon_end', 0.05)     # Minimum exploration rate (5% final)
        self.epsilon_decay = config.get('epsilon_decay', 0.995) # How fast to reduce exploration
        
        # Neural Networks for Q-Learning
        # Main Q-network: learns Q-values from observations
        self.q_network = QNetwork(obs_space, action_space, config).to(self.device)
        
        # Target Q-network: stable version for computing target values
        # Deep copy ensures completely independent parameters
        self.target_q_network = copy.deepcopy(self.q_network)
        
        # Note: QMIX uses centralized learning, so individual agents don't have optimizers
        # The QMIX algorithm will handle all parameter updates centrally
        
        print(f"Initialized QMIX Agent {agent_id}")
        print(f"Q-network parameters: {sum(p.numel() for p in self.q_network.parameters())}")
        print(f"Starting epsilon: {self.epsilon}")
    
    def get_action(self, observation: Dict, training: bool = True) -> int:
        """
        Select an action using epsilon-greedy policy based on Q-values.
        
        This is the core decision-making method for QMIX agents. It balances
        exploration (trying new actions) with exploitation (using learned knowledge).
        
        Epsilon-Greedy Strategy:
        - With probability epsilon: choose random action (exploration)
        - With probability (1-epsilon): choose best action according to Q-network (exploitation)
        
        Args:
            observation (Dict): Current local observation for this agent
                               Example: {'image': grid_state, 'direction': facing_direction}
            training (bool): Whether agent is in training mode
                            True: Use exploration (epsilon-greedy)
                            False: Always choose best action (greedy)
            
        Returns:
            int: Selected action ID (0 to action_space-1)
                Example: 2 might represent "move left" in MultiGrid
        
        For MARL Beginners:
        This is the agent's "decision-making brain". Early in training, it explores
        randomly to discover what works. Later, it mostly uses what it has learned
        but still explores occasionally to avoid getting stuck.
        """
        # Check if we should explore (only during training)
        if training and np.random.random() < self.epsilon:
            # Exploration: choose a random action to discover new strategies
            action = np.random.randint(0, self.action_space)
            return action
        else:
            # Exploitation: choose the best action according to learned Q-values
            with torch.no_grad():  # Don't compute gradients for action selection (saves memory)
                # Convert observation to tensor format for neural network
                obs_tensor = self._process_observation(observation)
                
                # Get Q-values for all possible actions in this state
                q_values = self.q_network(obs_tensor)
                
                # Choose action with highest Q-value (greedy action)
                action = q_values.argmax(dim=-1).item()
                return action
    
    def get_q_values(self, observation: Dict) -> torch.Tensor:
        """
        Get Q-values for all actions given an observation.
        
        This method is used by the QMIX mixing network to get individual
        Q-values that will be combined into joint Q-values.
        
        Args:
            observation (Dict): Agent's local observation
            
        Returns:
            torch.Tensor: Q-values for all actions [batch_size, num_actions]
        
        For Beginners:
        This asks the agent "how good do you think each action is in this situation?"
        The mixing network will use these individual opinions to make team decisions.
        """
        obs_tensor = self._process_observation(observation)
        return self.q_network(obs_tensor)
    
    def get_target_q_values(self, observation: Dict) -> torch.Tensor:
        """
        Get target Q-values using the target network.
        
        Target networks provide stable target values for Q-learning updates.
        They are periodically updated with the main network's parameters.
        
        Args:
            observation (Dict): Agent's local observation
            
        Returns:
            torch.Tensor: Target Q-values for all actions
        
        For Beginners:
        This is like asking the agent's "previous version" what it thinks
        about each action. Using an older version as reference helps
        keep learning stable.
        """
        obs_tensor = self._process_observation(observation)
        return self.target_q_network(obs_tensor)
    
    def update_target_network(self):
        """
        Update target network by copying parameters from main network.
        
        This synchronizes the target network with the current main network,
        providing updated but stable targets for Q-learning.
        
        For Beginners:
        This is like updating your "reference book" with your current knowledge.
        It happens periodically to keep the reference current but not too frequently
        to maintain stability.
        """
        self.target_q_network.load_state_dict(self.q_network.state_dict())
    
    def decay_epsilon(self):
        """
        Decay the exploration rate (epsilon) over time.
        
        As the agent learns more, it should explore less and exploit more.
        This gradually reduces random exploration in favor of learned behavior.
        
        For Beginners:
        Think of this as becoming more confident over time. A beginner tries
        many random things, but an expert mostly uses what they know works.
        """
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
    
    def _process_observation(self, observation: Dict) -> torch.Tensor:
        """
        Convert observation dictionary to tensor format for neural networks.
        
        Neural networks require numerical tensor inputs, so this method converts
        the environment's observation format into what the network expects.
        
        Args:
            observation (Dict): Raw observation from environment
                               Can contain images, vectors, scalars, etc.
            
        Returns:
            torch.Tensor: Processed observation ready for neural network
        
        For Beginners:
        This is like translating the environment's "language" into the
        neural network's "language". Different environments give observations
        in different formats, but neural networks need consistent tensor input.
        """
        if isinstance(observation, dict):
            # Handle dictionary observations (common in complex environments)
            obs_list = []
            for key, value in observation.items():
                if isinstance(value, np.ndarray):
                    # Convert numpy arrays to tensors and flatten
                    obs_list.append(torch.tensor(value, dtype=torch.float32).flatten())
                else:
                    # Convert scalars to single-element tensors
                    obs_list.append(torch.tensor([value], dtype=torch.float32))
            
            # Concatenate all observation components into single tensor
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
        """QMIX uses centralized replay buffer."""
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
        torch.save(checkpoint, f"{path}_qmix_agent_{self.agent_id}.pth")
        print(f"Saved QMIX Agent {self.agent_id} model to {path}_qmix_agent_{self.agent_id}.pth")
    
    def load_model(self, path: str):
        """Load the agent's model from disk."""
        checkpoint = torch.load(f"{path}_qmix_agent_{self.agent_id}.pth", map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_q_network.load_state_dict(checkpoint['target_q_network_state_dict'])
        self.epsilon = checkpoint['epsilon']
        print(f"Loaded QMIX Agent {self.agent_id} model from {path}_qmix_agent_{self.agent_id}.pth")


class QNetwork(nn.Module):
    """Q-network for individual agents."""
    
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


class MixingNetwork(nn.Module):
    """
    Mixing network that combines individual Q-values into joint Q-value.
    
    Ensures that the joint Q-value is monotonic in each agent's Q-value,
    satisfying the Individual-Global-Max (IGM) principle.
    """
    
    def __init__(self, n_agents: int, state_dim: int, config: Dict):
        super().__init__()
        
        self.n_agents = n_agents
        self.state_dim = state_dim
        
        # Hypernetwork dimensions
        self.embed_dim = config.get('mixing_embed_dim', 32)
        self.hypernet_embed = config.get('hypernet_embed_dim', 64)
        
        # Hypernetworks for generating weights and biases
        # First layer
        self.hyper_w_1 = nn.Linear(state_dim, self.hypernet_embed)
        self.hyper_w_final = nn.Linear(self.hypernet_embed, self.embed_dim * n_agents)
        
        # Second layer
        self.hyper_w_2 = nn.Linear(state_dim, self.hypernet_embed)
        self.hyper_w_final_2 = nn.Linear(self.hypernet_embed, self.embed_dim)
        
        # State-dependent bias
        self.hyper_b_1 = nn.Linear(state_dim, self.embed_dim)
        self.hyper_b_2 = nn.Sequential(
            nn.Linear(state_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1)
        )
    
    def forward(self, agent_qs: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through mixing network.
        
        Args:
            agent_qs: Individual Q-values [batch_size, n_agents]
            states: Global state information [batch_size, state_dim]
            
        Returns:
            Mixed Q-value [batch_size, 1]
        """
        batch_size = agent_qs.size(0)
        
        # First layer
        w1 = torch.abs(self.hyper_w_final(F.relu(self.hyper_w_1(states))))
        w1 = w1.view(batch_size, self.n_agents, self.embed_dim)
        
        b1 = self.hyper_b_1(states)
        b1 = b1.view(batch_size, 1, self.embed_dim)
        
        # agent_qs: [batch_size, n_agents] -> [batch_size, 1, n_agents]
        agent_qs = agent_qs.unsqueeze(1)
        
        # First layer computation: [batch_size, 1, n_agents] * [batch_size, n_agents, embed_dim]
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)
        
        # Second layer
        w2 = torch.abs(self.hyper_w_final_2(F.relu(self.hyper_w_2(states))))
        w2 = w2.view(batch_size, self.embed_dim, 1)
        
        b2 = self.hyper_b_2(states)
        
        # Second layer computation: [batch_size, 1, embed_dim] * [batch_size, embed_dim, 1]
        q_tot = torch.bmm(hidden, w2) + b2
        
        return q_tot.squeeze()


class QMIX(MARLAlgorithm):
    """
    QMIX algorithm implementation.
    
    Combines individual Q-learning with a mixing network for joint action-value estimation.
    """
    
    def __init__(self, env, config: Dict, device: torch.device):
        """
        Initialize the QMIX algorithm.
        
        Args:
            env: Multi-agent environment
            config: Configuration dictionary
            device: PyTorch device
        """
        super().__init__(env, config, device)
        
        # QMIX specific parameters
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
        self.replay_buffer = EpisodeReplayBuffer(
            self.buffer_size, obs_shape, self.n_agents, config.get('max_episode_length', 100)
        )
        
        # Create mixing network
        state_dim = self._get_state_dim()
        self.mixing_network = MixingNetwork(self.n_agents, state_dim, config).to(device)
        self.target_mixing_network = copy.deepcopy(self.mixing_network)
        
        # Optimizer for all networks
        params = list(self.mixing_network.parameters())
        for agent in self.agents:
            params.extend(list(agent.q_network.parameters()))
        
        self.optimizer = Adam(params, lr=self.lr)
        
        self.steps_done = 0
        
        print(f"Initialized QMIX with mixing network")
        print(f"Mixing network parameters: {sum(p.numel() for p in self.mixing_network.parameters())}")
    
    def _create_agents(self):
        """Create QMIX agents."""
        obs_space = self._get_obs_space()
        action_space = len(self.env.actions)
        
        for i in range(self.n_agents):
            agent = QMIXAgent(i, obs_space, action_space, self.config)
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
    
    def _get_state_dim(self) -> int:
        """Get global state dimension."""
        # For now, use concatenated observations as global state
        return self._get_obs_shape()[0] * self.n_agents
    
    def collect_rollout(self, env) -> Dict:
        """Collect a full episode for QMIX training."""
        episode_obs = []
        episode_actions = []
        episode_rewards = []
        episode_dones = []
        episode_states = []
        
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        while not done:
            # Store current state
            episode_obs.append(obs)
            state = self._get_global_state(obs)
            episode_states.append(state)
            
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
            'dones': episode_dones,
            'states': episode_states
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
    
    def _get_global_state(self, obs) -> np.ndarray:
        """Get global state representation."""
        # Simple implementation: concatenate all agent observations
        if isinstance(obs, dict):
            state_parts = []
            for i in range(self.n_agents):
                agent_obs = self._extract_agent_obs(obs, i)
                # Flatten agent observation
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
    
    def train_step(self, rollout_data: Dict) -> Dict[str, float]:
        """Perform QMIX training step."""
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
        """Update Q-networks and mixing network."""
        # Sample batch of episodes
        batch = self.replay_buffer.sample(self.batch_size)
        
        # Move to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(self.device)
        
        # Compute current Q-values
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
        
        current_q_values = torch.stack(current_q_values).transpose(0, 1)  # [episode_length, batch_size, n_agents]
        
        # Compute target Q-values
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
        
        target_q_values = torch.stack(target_q_values).transpose(0, 1)  # [episode_length, batch_size, n_agents]
        
        # Compute mixed Q-values
        mixed_q_values = []
        target_mixed_q_values = []
        
        for t in range(batch['episode_length']):
            state_t = batch['states'][t]  # [batch_size, state_dim]
            
            # Current mixed Q-values
            q_tot = self.mixing_network(current_q_values[t], state_t)
            mixed_q_values.append(q_tot)
            
            # Target mixed Q-values
            target_q_tot = self.target_mixing_network(target_q_values[t], state_t)
            target_mixed_q_values.append(target_q_tot)
        
        mixed_q_values = torch.stack(mixed_q_values)  # [episode_length, batch_size]
        target_mixed_q_values = torch.stack(target_mixed_q_values)  # [episode_length, batch_size]
        
        # Compute loss
        loss = F.mse_loss(mixed_q_values, target_mixed_q_values.detach())
        
        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            list(self.mixing_network.parameters()) + 
            [p for agent in self.agents for p in agent.q_network.parameters()],
            self.grad_norm_clip
        )
        
        self.optimizer.step()
        
        return loss.item()
    
    def _update_target_networks(self):
        """Update target networks."""
        # Update mixing network target
        self.target_mixing_network.load_state_dict(self.mixing_network.state_dict())
        
        # Update agent target networks
        for agent in self.agents:
            agent.update_target_network()


class EpisodeReplayBuffer:
    """
    Episode-based replay buffer for QMIX.
    
    Stores full episodes rather than individual transitions.
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
                episode['states'].append(episode['states'][-1])  # Repeat last state
        
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
            'states': [],
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
                    batch['states'].append([])
                
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
                batch['states'][t].append(episode['states'][t])
        
        # Convert to tensors
        for t in range(self.max_episode_length):
            batch['observations'][t] = torch.FloatTensor(batch['observations'][t])  # [batch_size, n_agents, obs_dim]
            batch['actions'][t] = torch.LongTensor(batch['actions'][t])  # [batch_size, n_agents]
            batch['rewards'][t] = torch.FloatTensor(batch['rewards'][t])  # [batch_size, n_agents]
            batch['dones'][t] = torch.FloatTensor(batch['dones'][t])  # [batch_size, n_agents]
            batch['states'][t] = torch.FloatTensor(batch['states'][t])  # [batch_size, state_dim]
        
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
