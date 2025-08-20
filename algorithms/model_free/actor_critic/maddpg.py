"""
(C) Shreyan Mitra, based on starter code by Natasha Jaques

Multi-Agent Deep Deterministic Policy Gradient (MADDPG) Algorithm

MADDPG is a sophisticated multi-agent reinforcement learning algorithm that addresses
one of the key challenges in MARL: how to train agents that can coordinate effectively
while still being able to act independently during execution.

Key Innovation - Centralized Training, Decentralized Execution (CTDE):
- Training: Agents can see all other agents' observations and actions (centralized)
- Execution: Each agent acts based only on its local observations (decentralized)
- This gives the best of both worlds: coordination during learning, independence during deployment

How MADDPG Works:
1. Each agent has an actor network (policy) and a critic network (Q-function)
2. Actor networks only see local observations (for decentralized execution)
3. Critic networks see global information (all agents' observations and actions)
4. Uses experience replay and target networks for stable learning
5. Handles both continuous and discrete action spaces

When to Use MADDPG:
✅ Mixed cooperative-competitive environments
✅ Need for sophisticated coordination
✅ Continuous or discrete action spaces
✅ When centralized training is possible but decentralized execution is required
✅ Environments with partial observability

Comparison with Other Algorithms:
- vs IPPO: More coordination but more complex
- vs QMIX: Works with continuous actions, handles mixed scenarios better
- vs MAPPO: Off-policy (can reuse old data) vs on-policy

For MARL Beginners:
MADDPG is more advanced than IPPO or QMIX. Try those first, then come to MADDPG
when you need more sophisticated coordination or mixed competitive-cooperative scenarios.

Paper: "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments" (2017)
Use Cases: Robotic coordination, autonomous vehicles, financial trading, competitive games
"""

# Import necessary libraries for deep learning and multi-agent systems
import torch                    # PyTorch for neural networks
import torch.nn as nn           # Neural network modules
import torch.nn.functional as F # Activation functions and utilities
from torch.optim import Adam    # Adam optimizer for gradient-based learning
import numpy as np              # Numerical computations
from typing import Dict, List, Tuple, Any  # Type hints for code clarity
import copy                     # For creating deep copies of networks

# Import base classes and network architectures from our MARL framework
from .base import MARLAgent, MARLAlgorithm, ReplayBuffer
from networks.multigrid_network import MultiGridNetwork


class MADDPGAgent(MARLAgent):
    """
    Multi-Agent Deep Deterministic Policy Gradient Agent.
    
    This agent implements the MADDPG algorithm, which extends the single-agent DDPG
    algorithm to multi-agent settings. The key insight is using centralized critics
    that can see global information during training, while keeping actors decentralized.
    
    Architecture:
    1. Actor Network: π(action|local_observation) - only sees local info
    2. Critic Network: Q(global_state, all_actions) - sees everything during training
    3. Target Networks: Stable versions of both networks for training stability
    
    Key Features:
    - Centralized Training, Decentralized Execution (CTDE)
    - Handles both discrete and continuous action spaces
    - Uses experience replay for sample efficiency
    - Soft target updates for training stability
    
    For MARL Beginners:
    Think of this as having a coach (centralized critic) who can see the whole field
    during practice, but players (decentralized actors) who can only see their local
    area during the actual game.
    """
    
    def __init__(self, agent_id: int, obs_space: Dict, action_space: int, 
                 n_agents: int, config: Dict):
        """
        Initialize the MADDPG agent with actor and critic networks.
        
        Args:
            agent_id (int): Unique identifier for this agent
            obs_space (Dict): Local observation space for this agent
                             Example: {'image': (7, 7, 3), 'direction': 4}
            action_space (int): Number of actions this agent can take
                               Example: 6 for discrete actions, or continuous dimension
            n_agents (int): Total number of agents in the environment
                           Needed for critic network that sees all agents
            config (Dict): Configuration containing hyperparameters
                          Example: {'gamma': 0.99, 'tau': 0.01, 'lr_actor': 1e-4, ...}
        
        For Beginners:
        This sets up one agent with both local decision-making ability (actor)
        and global situation awareness during training (critic).
        """
        # Call parent class constructor to set up basic agent properties
        super().__init__(agent_id, obs_space, action_space, config)
        
        # Store number of agents (needed for centralized critic)
        self.n_agents = n_agents
        
        # MADDPG Core Hyperparameters
        # Discount factor: how much the agent values future rewards
        self.gamma = config.get('gamma', 0.99)
        
        # Soft update parameter: how quickly to update target networks
        # tau=0.01 means target = 0.99*old_target + 0.01*current_network
        self.tau = config.get('tau', 0.01)
        
        # Exploration noise: random noise added to actions during training
        # Helps agent explore different strategies
        self.exploration_noise = config.get('exploration_noise', 0.1)
        
        # Learning rates: separate rates for actor and critic networks
        self.lr_actor = config.get('lr_actor', 1e-4)    # Actor learns slower (more stable)
        self.lr_critic = config.get('lr_critic', 1e-3)  # Critic learns faster
        
        # For discrete action spaces: use Gumbel-Softmax for differentiable sampling
        # Temperature controls how "sharp" the probability distribution is
        self.temperature = config.get('temperature', 1.0)
        self.hard_gumbel = config.get('hard_gumbel', True)
        
        # Networks
        self.actor = MADDPGActor(obs_space, action_space, config).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        
        # Critic takes global state and all actions
        self.critic = MADDPGCritic(obs_space, action_space, n_agents, config).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        
        # Optimizers
        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.lr_critic)
        
        print(f"Initialized MADDPG Agent {agent_id}")
        print(f"Actor parameters: {sum(p.numel() for p in self.actor.parameters())}")
        print(f"Critic parameters: {sum(p.numel() for p in self.critic.parameters())}")
    
    def get_action(self, observation: Dict, training: bool = True) -> np.ndarray:
        """
        Select an action given the current observation.
        
        Args:
            observation: Current observation from the environment
            training: Whether the agent is in training mode
            
        Returns:
            Action probabilities (for discrete actions)
        """
        with torch.no_grad():
            obs_tensor = self._process_observation(observation)
            action_probs = self.actor(obs_tensor)
            
            if training:
                # Add exploration noise using Gumbel-Softmax
                action_probs = F.gumbel_softmax(
                    action_probs, tau=self.temperature, hard=self.hard_gumbel
                )
            else:
                # Use greedy action during evaluation
                action_probs = F.softmax(action_probs, dim=-1)
        
        return action_probs.cpu().numpy().squeeze()
    
    def get_action_logits(self, observation: Dict) -> torch.Tensor:
        """Get raw action logits (used for training)."""
        obs_tensor = self._process_observation(observation)
        return self.actor(obs_tensor)
    
    def update_critic(self, batch: Dict, other_agents: List['MADDPGAgent']) -> float:
        """
        Update the critic network.
        
        Args:
            batch: Batch of experiences
            other_agents: List of other agents for centralized training
            
        Returns:
            Critic loss value
        """
        batch_size = batch['observations'].shape[0]
        
        # Current Q values
        # Collect all current observations and actions
        all_obs = []
        all_actions = []
        
        for i in range(self.n_agents):
            if i == self.agent_id:
                obs_i = batch['observations'][:, i]
                actions_i = batch['actions'][:, i]
            else:
                # Use other agents' data
                obs_i = batch['observations'][:, i]
                actions_i = batch['actions'][:, i]
            
            all_obs.append(obs_i)
            all_actions.append(actions_i)
        
        # Stack observations and actions
        global_obs = torch.cat([obs.reshape(batch_size, -1) for obs in all_obs], dim=1)
        global_actions = torch.cat(all_actions, dim=1)
        
        current_q = self.critic(global_obs, global_actions)
        
        # Target Q values
        with torch.no_grad():
            # Get next actions from target actors
            next_actions = []
            
            for i in range(self.n_agents):
                if i == self.agent_id:
                    next_obs_i = batch['next_observations'][:, i]
                    next_action_logits = self.actor_target(next_obs_i.reshape(batch_size, -1))
                else:
                    # Use other agents' target actors
                    next_obs_i = batch['next_observations'][:, i]
                    next_action_logits = other_agents[i].actor_target(next_obs_i.reshape(batch_size, -1))
                
                next_action_probs = F.gumbel_softmax(
                    next_action_logits, tau=self.temperature, hard=self.hard_gumbel
                )
                next_actions.append(next_action_probs)
            
            next_global_obs = torch.cat([obs.reshape(batch_size, -1) for obs in batch['next_observations'].unbind(1)], dim=1)
            next_global_actions = torch.cat(next_actions, dim=1)
            
            next_q = self.critic_target(next_global_obs, next_global_actions)
            target_q = batch['rewards'][:, self.agent_id].unsqueeze(1) + \
                      self.gamma * next_q * (1 - batch['dones'][:, self.agent_id].unsqueeze(1))
        
        # Critic loss
        critic_loss = F.mse_loss(current_q, target_q)
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        return critic_loss.item()
    
    def update_actor(self, batch: Dict, other_agents: List['MADDPGAgent']) -> float:
        """
        Update the actor network.
        
        Args:
            batch: Batch of experiences
            other_agents: List of other agents
            
        Returns:
            Actor loss value
        """
        batch_size = batch['observations'].shape[0]
        
        # Get current actions from all agents
        all_obs = []
        all_actions = []
        
        for i in range(self.n_agents):
            obs_i = batch['observations'][:, i]
            all_obs.append(obs_i)
            
            if i == self.agent_id:
                # Use current actor for this agent
                action_logits = self.actor(obs_i.reshape(batch_size, -1))
                action_probs = F.gumbel_softmax(
                    action_logits, tau=self.temperature, hard=self.hard_gumbel
                )
            else:
                # Use other agents' actions (detached)
                action_probs = batch['actions'][:, i].detach()
            
            all_actions.append(action_probs)
        
        # Stack for global state-action
        global_obs = torch.cat([obs.reshape(batch_size, -1) for obs in all_obs], dim=1)
        global_actions = torch.cat(all_actions, dim=1)
        
        # Actor loss (maximize Q value)
        q_value = self.critic(global_obs, global_actions)
        actor_loss = -q_value.mean()
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        return actor_loss.item()
    
    def soft_update(self):
        """Soft update of target networks."""
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):\n            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):\n            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
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
    
    def reset_memory(self):
        """MADDPG uses centralized replay buffer, so individual agents don't store memory."""
        pass
    
    def save_model(self, path: str):
        """Save the agent's model to disk."""
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'agent_id': self.agent_id,
            'config': self.config
        }
        torch.save(checkpoint, f"{path}_maddpg_agent_{self.agent_id}.pth")
        print(f"Saved MADDPG Agent {self.agent_id} model to {path}_maddpg_agent_{self.agent_id}.pth")
    
    def load_model(self, path: str):
        """Load the agent's model from disk."""
        checkpoint = torch.load(f"{path}_maddpg_agent_{self.agent_id}.pth", map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        print(f"Loaded MADDPG Agent {self.agent_id} model from {path}_maddpg_agent_{self.agent_id}.pth")


class MADDPGActor(nn.Module):
    """Actor network for MADDPG agent."""
    
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
        """Forward pass through actor network."""
        return self.network(obs)


class MADDPGCritic(nn.Module):
    """Critic network for MADDPG agent (centralized)."""
    
    def __init__(self, obs_space: Dict, action_space: int, n_agents: int, config: Dict):
        super().__init__()
        
        # Estimate observation dimension per agent
        self.obs_dim_per_agent = self._estimate_obs_dim(obs_space)
        self.global_obs_dim = self.obs_dim_per_agent * n_agents
        self.global_action_dim = action_space * n_agents
        
        hidden_dim = config.get('hidden_dim', 128)
        
        # State processing
        self.obs_encoder = nn.Sequential(
            nn.Linear(self.global_obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Action processing
        self.action_encoder = nn.Sequential(
            nn.Linear(self.global_action_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Q-value head
        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
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
                total_dim += 1  # Scalar value
        return total_dim
    
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, global_obs: torch.Tensor, global_actions: torch.Tensor) -> torch.Tensor:
        """Forward pass through critic network."""
        obs_features = self.obs_encoder(global_obs)
        action_features = self.action_encoder(global_actions)
        
        combined_features = torch.cat([obs_features, action_features], dim=-1)
        q_value = self.q_head(combined_features)
        
        return q_value


class MADDPG(MARLAlgorithm):
    """
    Multi-Agent Deep Deterministic Policy Gradient (MADDPG) algorithm.
    
    Uses centralized training with decentralized execution.
    """
    
    def __init__(self, env, config: Dict, device: torch.device):
        """
        Initialize the MADDPG algorithm.
        
        Args:
            env: Multi-agent environment
            config: Configuration dictionary
            device: PyTorch device
        """
        super().__init__(env, config, device)
        
        # MADDPG specific parameters
        self.batch_size = config.get('batch_size', 64)
        self.buffer_size = config.get('buffer_size', 100000)
        self.learning_starts = config.get('learning_starts', 1000)
        self.update_freq = config.get('update_freq', 1)
        self.target_update_freq = config.get('target_update_freq', 1)
        
        # Create centralized replay buffer
        obs_shape = self._get_obs_shape()
        self.replay_buffer = ReplayBuffer(
            self.buffer_size, obs_shape, self.env.action_space.n, self.n_agents
        )
        
        self.steps_done = 0
        
        print(f"Initialized MADDPG with centralized replay buffer")
        print(f"Buffer size: {self.buffer_size}, Batch size: {self.batch_size}")
    
    def _create_agents(self):
        """Create MADDPG agents."""
        obs_space = self._get_obs_space()
        action_space = len(self.env.actions)
        
        for i in range(self.n_agents):
            agent = MADDPGAgent(i, obs_space, action_space, self.n_agents, self.config)
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
        """Get the shape of flattened observations."""
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
        """
        Collect experiences and store them in the replay buffer.
        """
        obs = env.reset()
        done = False
        episode_rewards = [0.0] * self.n_agents
        episode_length = 0
        
        while not done:
            # Get actions from all agents
            actions = []
            action_probs = []
            
            for i, agent in enumerate(self.agents):
                # Extract agent observation
                if isinstance(obs, dict):
                    agent_obs = {}
                    for key, value in obs.items():
                        if isinstance(value, list):
                            agent_obs[key] = value[i] if i < len(value) else value[0]
                        else:
                            agent_obs[key] = value
                else:
                    agent_obs = obs[i] if isinstance(obs, list) else obs
                
                # Get action probabilities
                action_prob = agent.get_action(agent_obs, training=True)
                action_probs.append(action_prob)
                
                # Convert to discrete action
                action = np.argmax(action_prob)
                actions.append(action)
            
            # Take environment step
            next_obs, rewards, done, info = env.step(actions)
            
            # Store in replay buffer
            self._store_transition(obs, action_probs, rewards, next_obs, done)
            
            # Update rewards and obs
            if isinstance(rewards, list):
                for i in range(self.n_agents):
                    episode_rewards[i] += rewards[i]
            else:
                for i in range(self.n_agents):
                    episode_rewards[i] += rewards
            
            obs = next_obs
            episode_length += 1
            self.steps_done += 1
        
        return {
            'episode_rewards': episode_rewards,
            'episode_length': episode_length
        }
    
    def _store_transition(self, obs, actions, rewards, next_obs, done):
        """Store a transition in the replay buffer."""
        # Convert observations to arrays
        obs_array = self._obs_to_array(obs)
        next_obs_array = self._obs_to_array(next_obs)
        
        # Convert actions to arrays
        actions_array = np.array(actions)
        
        # Convert rewards
        if not isinstance(rewards, list):
            rewards = [rewards] * self.n_agents
        rewards_array = np.array(rewards)
        
        # Convert done
        done_array = np.array([done] * self.n_agents)
        
        self.replay_buffer.add(obs_array, actions_array, rewards_array, next_obs_array, done_array)
    
    def _obs_to_array(self, obs) -> np.ndarray:
        """Convert observation to numpy array format."""
        if isinstance(obs, dict):
            obs_arrays = []
            for i in range(self.n_agents):
                agent_obs = []
                for key, value in obs.items():
                    if isinstance(value, list):
                        agent_value = value[i] if i < len(value) else value[0]
                    else:
                        agent_value = value
                    
                    if isinstance(agent_value, np.ndarray):
                        agent_obs.append(agent_value.flatten())
                    else:
                        agent_obs.append(np.array([agent_value]))
                
                obs_arrays.append(np.concatenate(agent_obs))
            
            return np.array(obs_arrays)
        else:
            return np.array(obs)
    
    def train_step(self, rollout_data: Dict) -> Dict[str, float]:
        """Perform training step."""
        metrics = {
            'episode_reward': np.mean(rollout_data['episode_rewards']),
            'episode_length': rollout_data['episode_length']
        }
        
        # Only train if we have enough samples
        if len(self.replay_buffer) < self.learning_starts:
            return metrics
        
        # Train agents
        if self.steps_done % self.update_freq == 0:
            # Sample batch
            batch = self.replay_buffer.sample(self.batch_size)
            
            # Move to device
            for key in batch:
                batch[key] = batch[key].to(self.device)
            
            # Update critics first
            critic_losses = []
            for agent in self.agents:
                critic_loss = agent.update_critic(batch, self.agents)
                critic_losses.append(critic_loss)
            
            # Update actors
            actor_losses = []
            for agent in self.agents:
                actor_loss = agent.update_actor(batch, self.agents)
                actor_losses.append(actor_loss)
            
            # Soft update target networks
            if self.steps_done % self.target_update_freq == 0:
                for agent in self.agents:
                    agent.soft_update()
            
            metrics.update({
                'mean_critic_loss': np.mean(critic_losses),
                'mean_actor_loss': np.mean(actor_losses),
                'buffer_size': len(self.replay_buffer)
            })
        
        return metrics
