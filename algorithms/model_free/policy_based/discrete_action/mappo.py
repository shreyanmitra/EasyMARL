"""
(C) Shreyan Mitra, based on starter code by Natasha Jaques

Multi-Agent Proximal Policy Optimization (MAPPO) Algorithm

MAPPO is the state-of-the-art policy gradient method for cooperative multi-agent
reinforcement learning. It extends PPO to multi-agent settings by using centralized
training with decentralized execution (CTDE) paradigm.

Key Innovation - Centralized Value Functions:
- Training: Value functions can see global state (centralized)
- Execution: Policies only use local observations (decentralized)
- This enables better credit assignment while maintaining practical deployment

How MAPPO Works:
1. Each agent has a decentralized policy π(action|local_observation)
2. All agents share a centralized value function V(global_state)
3. Uses PPO's clipping mechanism for stable policy updates
4. Parameter sharing across agents for improved sample efficiency
5. Generalized Advantage Estimation (GAE) for better gradient estimates

When to Use MAPPO:
✅ Cooperative multi-agent tasks (agents work toward common goal)
✅ Need for sophisticated coordination
✅ Large number of agents (parameter sharing helps)
✅ Continuous or discrete action spaces
✅ When sample efficiency is important

Key Advantages:
✅ State-of-the-art performance on many benchmarks
✅ Stable training due to PPO's clipping mechanism
✅ Scalable to many agents through parameter sharing
✅ Handles both discrete and continuous actions
✅ Strong theoretical foundations

Comparison with Other Algorithms:
- vs IPPO: Centralized training provides better coordination
- vs QMIX: Policy gradients vs value-based, handles continuous actions
- vs MADDPG: More stable training, better for cooperative tasks

For MARL Beginners:
MAPPO is advanced but very important. Start with IPPO to understand PPO,
then move to MAPPO to see how centralized training improves coordination.
It's currently the gold standard for cooperative MARL.

Paper: "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games" (2021)
Use Cases: Cooperative robotics, team coordination, resource management, StarCraft II
"""

# Import necessary libraries for deep learning and multi-agent systems
import torch                    # PyTorch for neural networks
import torch.nn as nn           # Neural network modules
import torch.nn.functional as F # Activation functions and utilities
from torch.distributions import Categorical  # For sampling from probability distributions
from torch.optim import Adam    # Adam optimizer for gradient-based learning
import numpy as np              # Numerical computations
from typing import Dict, List, Tuple, Any  # Type hints for code clarity

# Import base classes and utilities from our MARL framework
from .base import MARLAgent, MARLAlgorithm, compute_gae, normalize_advantages
from networks.multigrid_network import MultiGridNetwork


class MAPPOAgent(MARLAgent):
    """
    Multi-Agent Proximal Policy Optimization Agent.
    
    This agent implements MAPPO, which combines the stability of PPO with the
    coordination benefits of centralized training. Each agent has a decentralized
    policy for execution but uses a centralized value function during training.
    
    Key Components:
    1. Decentralized Policy: π(action|local_observation) - only sees local info
    2. Centralized Critic: V(global_state) - sees global information during training
    3. PPO Clipping: Prevents large policy updates for stable learning
    4. Parameter Sharing: Agents can share parameters for better sample efficiency
    
    Architecture Insight:
    - Policy Network: Input = local observation → Output = action probabilities
    - Value Network: Input = global state → Output = state value estimate
    - Shared Parameters: Multiple agents can use the same networks (optional)
    
    For MARL Beginners:
    Think of this as IPPO agents with a "shared coach" (centralized critic) who
    can see the whole field during training, but players (policies) still only
    see their local area during the game.
    """
    
    def __init__(self, agent_id: int, obs_space: Dict, action_space: int, 
                 global_state_dim: int, n_agents: int, config: Dict):
        """
        Initialize the MAPPO agent with policy and value networks.
        
        Args:
            agent_id (int): Unique identifier for this agent
            obs_space (Dict): Local observation space for this agent
                             Example: {'image': (7, 7, 3), 'direction': 4}
            action_space (int): Number of actions this agent can take
                               Example: 6 for MultiGrid environments
            global_state_dim (int): Dimension of global state for centralized critic
                                   This includes all agents' observations
            n_agents (int): Total number of agents in the system
                           Used for parameter sharing decisions
            config (Dict): Configuration containing hyperparameters
                          Example: {'gamma': 0.99, 'clip_epsilon': 0.2, ...}
        
        For Beginners:
        This sets up an agent that can act independently but learns from
        global information during training.
        """
        # Call parent class constructor to set up basic agent properties
        super().__init__(agent_id, obs_space, action_space, config)
        
        # Store multi-agent specific information
        self.global_state_dim = global_state_dim  # Size of global state for critic
        self.n_agents = n_agents                  # Total number of agents
        
        # PPO Core Hyperparameters (same as IPPO but applied to multi-agent setting)
        # Discount factor: how much the agent values future rewards
        self.gamma = config.get('gamma', 0.99)
        
        # GAE lambda: controls bias-variance tradeoff in advantage estimation
        self.lambda_gae = config.get('lambda_gae', 0.95)
        
        # PPO clipping parameter: prevents too large policy updates
        self.clip_epsilon = config.get('clip_epsilon', 0.2)
        
        # Loss function coefficients: balance different learning objectives
        self.value_loss_coef = config.get('value_loss_coef', 0.5)    # Value function learning weight
        self.entropy_coef = config.get('entropy_coef', 0.01)         # Exploration encouragement
        
        # Gradient clipping: prevents exploding gradients
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        
        # Network Architecture Setup
        # Actor Network: Converts local observations to action probabilities
        # This is the "decentralized" part - only sees local information
        self.actor = MAPPOActor(obs_space, action_space, config).to(self.device)
        
        # Critic Network: Estimates value from global state (centralized training)
        # This is the "centralized" part - sees global information during training
        self.critic = MAPPOCritic(global_state_dim, config).to(self.device)
        
        # Optimizers for Learning
        # Separate optimizers allow different learning rates for actor and critic
        self.actor_optimizer = Adam(
            self.actor.parameters(), 
            lr=config.get('lr_actor', 3e-4),  # Actor learning rate (typically smaller)
            eps=1e-5                          # Numerical stability
        )
        self.critic_optimizer = Adam(
            self.critic.parameters(), 
            lr=config.get('lr_critic', 1e-3), # Critic learning rate (typically larger)
            eps=1e-5                           # Numerical stability
        )
        
        # Debug Information
        print(f"Initialized MAPPO Agent {agent_id}")
        print(f"Actor parameters: {sum(p.numel() for p in self.actor.parameters())}")
        print(f"Critic parameters: {sum(p.numel() for p in self.critic.parameters())}")
    
    def get_action(self, observation: Dict, global_state: np.ndarray, 
                   training: bool = True) -> Tuple[int, float, float]:
        """
        Select an action using the decentralized policy and estimate value using centralized critic.
        
        This method demonstrates the core MAPPO principle:
        - Policy (actor) only uses local observation (decentralized execution)
        - Value function (critic) uses global state (centralized training)
        
        Args:
            observation (Dict): Agent's local observation of the environment
                               Example: {'image': 7x7x3 grid, 'direction': int}
            global_state (np.ndarray): Complete global state for value estimation
                                      Includes all agents' observations and environment state
            training (bool): Whether agent is in training mode (affects action selection)
                           True: Sample actions from policy (exploration)
                           False: Take best action deterministically (exploitation)
            
        Returns:
            Tuple[int, float, float]: 
                - action: The selected action (integer)
                - log_probability: Log probability of selected action (for PPO updates)
                - value_estimate: Centralized critic's value estimate (for advantage calculation)
        
        For MARL Beginners:
        This is where MAPPO's "centralized training, decentralized execution" happens:
        1. Action selection only uses local observation (what the agent can see)
        2. Value estimation uses global state (what a central observer can see)
        3. During actual deployment, only the policy part is used
        """
        # Disable gradient computation for inference (saves memory and computation)
        with torch.no_grad():
            # Convert local observation to tensor format for neural network
            obs_tensor = self._process_observation(observation)
            
            # Convert global state to tensor for centralized critic
            state_tensor = torch.tensor(global_state, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # Get action probabilities from decentralized policy (actor)
            # This only uses LOCAL observation - key to decentralized execution
            action_logits = self.actor(obs_tensor)
            
            if training:
                # Training Mode: Sample action from probability distribution
                # This encourages exploration by sampling rather than always picking best action
                dist = Categorical(logits=action_logits)  # Create probability distribution
                action = dist.sample()                    # Sample an action
                log_prob = dist.log_prob(action)         # Get log probability for PPO updates
            else:
                # Evaluation Mode: Take the best action deterministically
                # This ensures consistent performance during testing
                action = torch.argmax(action_logits, dim=-1)  # Pick highest probability action
                dist = Categorical(logits=action_logits)      # Still need distribution for log_prob
                log_prob = dist.log_prob(action)             # Get log probability
            
            # Get value estimate from centralized critic
            # This uses GLOBAL state - key to centralized training advantage
            value = self.critic(state_tensor)
            
            # Return action, log probability, and value estimate
            return action.item(), log_prob.item(), value.item()
    
    def _process_observation(self, observation: Dict) -> torch.Tensor:
        """
        Convert observation dictionary to tensor format for neural network processing.
        
        This method handles the transformation from environment observations to
        the format expected by the actor network. MultiGrid environments typically
        provide observations as dictionaries with image data and additional features.
        
        Args:
            observation (Dict): Raw observation from environment
                               Example: {'image': np.array(7,7,3), 'direction': 2}
                               
        Returns:
            torch.Tensor: Flattened observation tensor ready for neural network
                         Shape: (1, total_observation_size)
        
        For MARL Beginners:
        This is a utility function that converts the environment's observation
        format into the format our neural networks expect (flat tensors).
        """
        if isinstance(observation, dict):
            # Process dictionary observations (common in MultiGrid)
            obs_list = []
            for key, value in observation.items():
                if isinstance(value, np.ndarray):
                    # Flatten multi-dimensional arrays (like images)
                    obs_list.append(torch.tensor(value, dtype=torch.float32).flatten())
                else:
                    # Convert single values to tensors
                    obs_list.append(torch.tensor([value], dtype=torch.float32))
            # Concatenate all observation components into a single tensor
            obs_tensor = torch.cat(obs_list).unsqueeze(0).to(self.device)
        else:
            # Handle simple array observations
            obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        return obs_tensor
    
    def store_transition(self, observation: Dict, global_state: np.ndarray, 
                        action: int, log_prob: float, value: float, 
                        reward: float, done: bool):
        """
        Store a single transition (experience) in the agent's memory buffer.
        
        This method collects the data needed for PPO updates. Each transition
        represents one step of interaction with the environment.
        
        Args:
            observation (Dict): Agent's local observation at this step
            global_state (np.ndarray): Global state for centralized value function
            action (int): Action taken by the agent
            log_prob (float): Log probability of the taken action
            value (float): Value estimate from centralized critic
            reward (float): Reward received from environment
            done (bool): Whether episode ended after this step
            
        For MARL Beginners:
        This is like keeping a diary of everything that happened during training.
        We store all this information so we can learn from it later using PPO.
        """
        # Store all components of the transition for later learning
        self.memory['observations'].append(observation)      # What the agent saw
        self.memory['global_states'].append(global_state)    # Global environment state
        self.memory['actions'].append(action)                # What the agent did
        self.memory['log_probs'].append(log_prob)           # How confident the agent was
        self.memory['values'].append(value)                  # How good the agent thought the state was
        self.memory['rewards'].append(reward)                # What reward the agent got
        self.memory['dones'].append(done)                    # Whether the episode ended
    
    def update_actor(self, batch_obs: List[Dict], batch_actions: torch.Tensor,
                     batch_old_log_probs: torch.Tensor, batch_advantages: torch.Tensor) -> float:
        """
        Update the actor (policy) network using PPO's clipped objective.
        
        This implements the core PPO policy update that prevents large policy changes
        while encouraging the agent to take actions that led to high advantages.
        
        Args:
            batch_obs (List[Dict]): Batch of observations from experience buffer
            batch_actions (torch.Tensor): Actions that were taken
            batch_old_log_probs (torch.Tensor): Log probabilities of actions under old policy
            batch_advantages (torch.Tensor): Advantage estimates (how much better actions were)
            
        Returns:
            float: Actor loss value for monitoring training progress
            
        For MARL Beginners:
        This is where the agent learns to improve its action selection based on
        which actions led to good outcomes (high advantages).
        """
        # Process batch of observations into tensor format
        obs_tensors = []
        for obs in batch_obs:
            obs_tensor = self._process_observation(obs)
            obs_tensors.append(obs_tensor)
        
        # Stack all observations into a single batch tensor
        stacked_obs = torch.cat(obs_tensors, dim=0)
        
        # Forward pass through current policy network
        action_logits = self.actor(stacked_obs)
        dist = Categorical(logits=action_logits)
        
        # Get log probabilities under current policy
        new_log_probs = dist.log_prob(batch_actions)
        entropy = dist.entropy()  # Measure of policy randomness (good for exploration)
        
        # PPO Clipped Policy Loss
        # This is the key innovation of PPO - prevents large policy updates
        ratio = torch.exp(new_log_probs - batch_old_log_probs)  # new_policy / old_policy
        surr1 = ratio * batch_advantages                        # Standard policy gradient
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages  # Clipped version
        policy_loss = -torch.min(surr1, surr2).mean()         # Take minimum (more conservative)
        
        # Entropy bonus encourages exploration
        entropy_loss = -entropy.mean()
        
        # Total actor loss combines policy improvement with exploration bonus
        actor_loss = policy_loss + self.entropy_coef * entropy_loss
        
        # Perform gradient descent update
        self.actor_optimizer.zero_grad()                                           # Clear previous gradients
        actor_loss.backward()                                                      # Compute gradients
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)  # Prevent exploding gradients
        self.actor_optimizer.step()                                                # Update parameters
        
        return actor_loss.item()
    
    def update_critic(self, batch_global_states: torch.Tensor, 
                     batch_returns: torch.Tensor) -> float:
        """
        Update the centralized critic network to better estimate state values.
        
        The critic learns to predict the expected return (sum of future rewards)
        from any given global state. This is crucial for advantage estimation.
        
        Args:
            batch_global_states (torch.Tensor): Global states from experience buffer
            batch_returns (torch.Tensor): Actual returns (discounted sum of rewards)
                                         These are the "ground truth" values to learn
            
        Returns:
            float: Critic loss value for monitoring training progress
            
        For MARL Beginners:
        The critic acts like a "coach" who learns to evaluate how good different
        game situations are by looking at the global state. This helps with
        calculating advantages for the actor updates.
        """
        # Forward pass through critic network
        values = self.critic(batch_global_states).squeeze()  # Get value predictions
        
        # Mean Squared Error loss between predicted and actual returns
        # This teaches the critic to accurately predict future rewards
        value_loss = F.mse_loss(values, batch_returns)
        
        # Perform gradient descent update
        self.critic_optimizer.zero_grad()                                             # Clear previous gradients
        value_loss.backward()                                                         # Compute gradients  
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm) # Prevent exploding gradients
        self.critic_optimizer.step()                                                  # Update parameters
        
        return value_loss.item()
    
    def reset_memory(self):
        """
        Clear the agent's experience memory buffer.
        
        This is called after each training update to prepare for collecting
        new experiences. MAPPO uses on-policy learning, so old experiences
        become stale after policy updates.
        
        For MARL Beginners:
        Think of this as clearing the agent's short-term memory after it
        has learned from recent experiences.
        """
        self.memory = {
            'observations': [],      # Agent's local observations
            'global_states': [],     # Global states for centralized critic
            'actions': [],           # Actions taken by agent
            'log_probs': [],        # Log probabilities of actions
            'values': [],           # Value estimates from critic
            'rewards': [],          # Rewards received from environment
            'dones': []             # Episode termination flags
        }
    
    def save_model(self, path: str):
        """
        Save the agent's neural networks and training state to disk.
        
        This saves both the actor and critic networks along with optimizer states,
        allowing training to be resumed later.
        
        Args:
            path (str): Base path for saving the model files
            
        For MARL Beginners:
        This lets you save your trained agent so you can use it later or
        continue training from where you left off.
        """
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),                    # Actor network weights
            'critic_state_dict': self.critic.state_dict(),                 # Critic network weights
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(), # Actor optimizer state
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(), # Critic optimizer state
            'agent_id': self.agent_id,                                      # Agent identifier
            'config': self.config                                           # Configuration used
        }
        torch.save(checkpoint, f"{path}_mappo_agent_{self.agent_id}.pth")
        print(f"Saved MAPPO Agent {self.agent_id} model to {path}_mappo_agent_{self.agent_id}.pth")
    
    def load_model(self, path: str):
        """
        Load a previously saved agent model from disk.
        
        This restores both neural networks and optimizer states, allowing
        training to continue from exactly where it was saved.
        
        Args:
            path (str): Base path where the model was saved
            
        For MARL Beginners:
        This loads a previously trained agent so you can continue training
        or use it for evaluation.
        """
        checkpoint = torch.load(f"{path}_mappo_agent_{self.agent_id}.pth", map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])           # Restore actor weights
        self.critic.load_state_dict(checkpoint['critic_state_dict'])         # Restore critic weights
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])   # Restore actor optimizer
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict']) # Restore critic optimizer
        print(f"Loaded MAPPO Agent {self.agent_id} model from {path}_mappo_agent_{self.agent_id}.pth")


class MAPPOActor(nn.Module):
    """Decentralized actor network for MAPPO."""
    
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
                total_dim += 1
        return total_dim
    
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass through actor."""
        return self.network(obs)


class MAPPOCritic(nn.Module):
    """Centralized critic network for MAPPO."""
    
    def __init__(self, global_state_dim: int, config: Dict):
        super().__init__()
        
        hidden_dim = config.get('hidden_dim', 128)
        
        self.network = nn.Sequential(
            nn.Linear(global_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        """Forward pass through critic."""
        return self.network(global_state)


class MAPPO(MARLAlgorithm):
    """
    Multi-Agent Proximal Policy Optimization with centralized critics.
    """
    
    def __init__(self, env, config: Dict, device: torch.device):
        """
        Initialize MAPPO algorithm.
        
        Args:
            env: Multi-agent environment
            config: Configuration dictionary
            device: PyTorch device
        """
        super().__init__(env, config, device)
        
        # MAPPO specific parameters
        self.rollout_length = config.get('rollout_length', 128)
        self.ppo_epochs = config.get('ppo_epochs', 4)
        self.mini_batch_size = config.get('mini_batch_size', 32)
        self.share_parameters = config.get('share_parameters', False)
        
        # Global state dimension
        self.global_state_dim = self._get_global_state_dim()
        
        print(f"Initialized MAPPO")
        print(f"Rollout length: {self.rollout_length}")
        print(f"PPO epochs: {self.ppo_epochs}")
        print(f"Parameter sharing: {self.share_parameters}")
        print(f"Global state dim: {self.global_state_dim}")
    
    def _create_agents(self):
        """Create MAPPO agents."""
        obs_space = self._get_obs_space()
        action_space = len(self.env.actions)
        
        if self.share_parameters:
            # Create one shared agent and copy for others
            shared_agent = MAPPOAgent(
                0, obs_space, action_space, self.global_state_dim, self.n_agents, self.config
            )
            self.agents.append(shared_agent)
            
            for i in range(1, self.n_agents):
                # Create agents that share parameters with the first one
                agent = MAPPOAgent(
                    i, obs_space, action_space, self.global_state_dim, self.n_agents, self.config
                )
                # Share parameters
                agent.actor = shared_agent.actor
                agent.critic = shared_agent.critic
                agent.actor_optimizer = shared_agent.actor_optimizer
                agent.critic_optimizer = shared_agent.critic_optimizer
                self.agents.append(agent)
            
            print("Using parameter sharing across agents")
        else:
            # Create independent agents
            for i in range(self.n_agents):
                agent = MAPPOAgent(
                    i, obs_space, action_space, self.global_state_dim, self.n_agents, self.config
                )
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
    
    def _get_global_state_dim(self) -> int:
        """Get dimension of global state."""
        # Use concatenated observations as global state
        obs_space = self._get_obs_space()
        obs_dim = 0
        for key, value in obs_space.items():
            if hasattr(value, 'shape'):
                obs_dim += np.prod(value.shape)
            elif isinstance(value, np.ndarray):
                obs_dim += np.prod(value.shape)
            else:
                obs_dim += 1
        
        return obs_dim * self.n_agents
    
    def _get_global_state(self, obs) -> np.ndarray:
        """Get global state from joint observations."""
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
        """Collect rollout for MAPPO training."""
        obs = env.reset()
        done = False
        step_count = 0
        
        # Initialize data storage
        episode_data = {
            'observations': [[] for _ in range(self.n_agents)],
            'global_states': [],
            'actions': [[] for _ in range(self.n_agents)],
            'log_probs': [[] for _ in range(self.n_agents)],
            'values': [[] for _ in range(self.n_agents)],
            'rewards': [[] for _ in range(self.n_agents)],
            'dones': [[] for _ in range(self.n_agents)]
        }
        
        while not done and step_count < self.rollout_length:
            # Get global state
            global_state = self._get_global_state(obs)
            episode_data['global_states'].append(global_state)
            
            # Get actions from all agents
            actions = []
            log_probs = []
            values = []
            
            for i, agent in enumerate(self.agents):
                agent_obs = self._extract_agent_obs(obs, i)
                action, log_prob, value = agent.get_action(agent_obs, global_state, training=True)
                
                actions.append(action)
                log_probs.append(log_prob)
                values.append(value)
                
                # Store in agent memory
                episode_data['observations'][i].append(agent_obs)
                episode_data['actions'][i].append(action)
                episode_data['log_probs'][i].append(log_prob)
                episode_data['values'][i].append(value)
            
            # Take environment step
            next_obs, rewards, done, info = env.step(actions)
            
            # Store rewards and dones
            if isinstance(rewards, list):
                for i in range(self.n_agents):
                    episode_data['rewards'][i].append(rewards[i])
                    episode_data['dones'][i].append(done)
            else:
                for i in range(self.n_agents):
                    episode_data['rewards'][i].append(rewards)
                    episode_data['dones'][i].append(done)
            
            obs = next_obs
            step_count += 1
            self.total_steps += 1
        
        # Get final values for bootstrapping
        if not done:
            final_global_state = self._get_global_state(obs)
            final_values = []
            for i, agent in enumerate(self.agents):
                agent_obs = self._extract_agent_obs(obs, i)
                _, _, value = agent.get_action(agent_obs, final_global_state, training=False)
                final_values.append(value)
        else:
            final_values = [0.0] * self.n_agents
        
        # Store transitions in agent memories
        for i, agent in enumerate(self.agents):
            agent.reset_memory()
            for t in range(step_count):
                agent.store_transition(
                    episode_data['observations'][i][t],
                    episode_data['global_states'][t],
                    episode_data['actions'][i][t],
                    episode_data['log_probs'][i][t],
                    episode_data['values'][i][t],
                    episode_data['rewards'][i][t],
                    episode_data['dones'][i][t]
                )
        
        return {
            'episode_data': episode_data,
            'final_values': final_values,
            'episode_length': step_count,
            'total_reward': [sum(episode_data['rewards'][i]) for i in range(self.n_agents)]
        }
    
    def train_step(self, rollout_data: Dict) -> Dict[str, float]:
        """Perform MAPPO training step."""
        episode_data = rollout_data['episode_data']
        final_values = rollout_data['final_values']
        
        # Compute advantages and returns for each agent
        all_advantages = []
        all_returns = []
        
        for i in range(self.n_agents):
            rewards = torch.tensor(episode_data['rewards'][i], dtype=torch.float32)
            values = torch.tensor(episode_data['values'][i], dtype=torch.float32)
            dones = torch.tensor(episode_data['dones'][i], dtype=torch.float32)
            
            # Bootstrap with final value
            next_values = torch.cat([values[1:], torch.tensor([final_values[i]])])
            
            # Compute GAE
            advantages = compute_gae(
                rewards.unsqueeze(1), values.unsqueeze(1), next_values.unsqueeze(1),
                dones.unsqueeze(1), self.gamma, self.lambda_gae
            ).squeeze(1)
            
            returns = advantages + values
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            all_advantages.append(advantages)
            all_returns.append(returns)
        
        # Training metrics
        total_actor_loss = 0
        total_critic_loss = 0
        
        # Multiple PPO epochs
        for epoch in range(self.ppo_epochs):
            # Update each agent (or shared parameters)
            if self.share_parameters:
                # Collect all data for shared training
                all_obs = []
                all_actions = []
                all_old_log_probs = []
                all_advs = []
                all_global_states = []
                all_rets = []
                
                for i in range(self.n_agents):
                    all_obs.extend(episode_data['observations'][i])
                    all_actions.extend(episode_data['actions'][i])
                    all_old_log_probs.extend(episode_data['log_probs'][i])
                    all_advs.extend(all_advantages[i].tolist())
                    all_rets.extend(all_returns[i].tolist())
                    all_global_states.extend(episode_data['global_states'])
                
                # Convert to tensors
                all_actions = torch.tensor(all_actions, dtype=torch.long)
                all_old_log_probs = torch.tensor(all_old_log_probs, dtype=torch.float32)
                all_advs = torch.tensor(all_advs, dtype=torch.float32)
                all_global_states = torch.tensor(all_global_states, dtype=torch.float32)
                all_rets = torch.tensor(all_rets, dtype=torch.float32)
                
                # Update shared networks
                actor_loss = self.agents[0].update_actor(
                    all_obs, all_actions, all_old_log_probs, all_advs
                )
                critic_loss = self.agents[0].update_critic(all_global_states, all_rets)
                
                total_actor_loss += actor_loss
                total_critic_loss += critic_loss
                
            else:
                # Update each agent independently
                for i, agent in enumerate(self.agents):
                    # Prepare data for this agent
                    agent_actions = torch.tensor(episode_data['actions'][i], dtype=torch.long)
                    agent_old_log_probs = torch.tensor(episode_data['log_probs'][i], dtype=torch.float32)
                    agent_global_states = torch.tensor(episode_data['global_states'], dtype=torch.float32)
                    
                    # Update networks
                    actor_loss = agent.update_actor(
                        episode_data['observations'][i], agent_actions, 
                        agent_old_log_probs, all_advantages[i]
                    )
                    critic_loss = agent.update_critic(agent_global_states, all_returns[i])
                    
                    total_actor_loss += actor_loss
                    total_critic_loss += critic_loss
        
        # Average losses
        num_updates = self.ppo_epochs * (1 if self.share_parameters else self.n_agents)
        
        return {
            'actor_loss': total_actor_loss / num_updates,
            'critic_loss': total_critic_loss / num_updates,
            'episode_length': rollout_data['episode_length'],
            'mean_episode_reward': np.mean(rollout_data['total_reward']),
            'total_steps': self.total_steps
        }
