"""
(C) Shreyan Mitra, based on starter code by Natasha Jaques

Independent Proximal Policy Optimization (IPPO) for Multi-Agent Reinforcement Learning

IPPO is one of the simplest and most effective MARL algorithms. It treats multi-agent
learning as multiple independent single-agent RL problems, where each agent learns
its own policy without explicitly coordinating with others.

Key Advantages of IPPO:
✅ Simple to understand and implement
✅ Stable training due to PPO's clipping mechanism
✅ No communication required between agents
✅ Scales well to many agents
✅ Works in both cooperative and competitive scenarios

How IPPO Works:
1. Each agent has its own actor network (policy) and critic network (value function)
2. Agents collect experiences independently from their local observations
3. Each agent updates its policy using standard PPO updates
4. No explicit coordination - agents adapt to each other through environment interaction

When to Use IPPO:
✅ First time learning MARL (excellent starting point)
✅ Large number of agents (scales better than centralized methods)
✅ Limited computational resources
✅ When agent coordination is not critical
✅ As a strong baseline to compare other algorithms against

Comparison with Other Algorithms:
- vs QMIX: Simpler but less coordinated
- vs MADDPG: More stable but less sample efficient
- vs MAPPO: Independent vs centralized training

For MARL Beginners:
Start with IPPO! It's the "Hello World" of multi-agent RL. Once you understand
IPPO, you can move to more sophisticated algorithms like QMIX or MADDPG.

Paper: "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments" (adapted for PPO)
Use Cases: Swarm robotics, traffic control, distributed optimization, competitive games
"""

# Try to import PyTorch and related libraries
try:
    import torch                          # Main PyTorch library for neural networks
    import torch.nn as nn                 # Neural network modules
    import torch.nn.functional as F       # Activation functions and utilities
    from torch.distributions import Categorical  # For sampling from probability distributions
    from torch.optim import Adam          # Adam optimizer for gradient-based learning
    TORCH_AVAILABLE = True
except ImportError:
    # Provide helpful message if PyTorch is not installed
    print("Warning: PyTorch not available. Install with: pip install torch")
    TORCH_AVAILABLE = False

# Import numerical computing and utility libraries
import numpy as np                        # Numerical computations and arrays
from typing import Dict, List, Tuple, Any  # Type hints for better code clarity

# Try to import Weights & Biases for experiment tracking (optional)
try:
    import wandb
except ImportError:
    wandb = None  # Will skip logging if not available

# Import base classes and utilities from our MARL framework
from .base import MARLAgent, MARLAlgorithm, compute_gae, normalize_advantages

# Import neural network architectures (only if PyTorch is available)
if TORCH_AVAILABLE:
    from networks.multigrid_network import MultiGridNetwork


class IPPOAgent(MARLAgent):
    """
    Independent Proximal Policy Optimization Agent.
    
    This agent implements the PPO algorithm independently for multi-agent environments.
    Each agent learns its own policy and value function without explicit coordination
    with other agents, but adapts to them through environment interactions.
    
    Key Components:
    1. Actor Network: Learns the policy π(action|observation)
    2. Critic Network: Learns the value function V(observation)
    3. PPO Clipping: Prevents large policy updates for stable training
    4. GAE: Generalized Advantage Estimation for better gradient estimates
    
    For MARL Beginners:
    Think of this as a single-agent PPO that happens to be in a multi-agent world.
    Each agent learns independently, like students studying for different subjects
    without directly helping each other, but still influenced by the classroom environment.
    """
    
    def __init__(self, agent_id: int, obs_space: Dict, action_space: int, config: Dict):
        """
        Initialize the IPPO agent with actor and critic networks.
        
        This sets up everything the agent needs to learn: its policy network (actor),
        value estimation network (critic), and learning hyperparameters.
        
        Args:
            agent_id (int): Unique identifier for this agent
            obs_space (Dict): What this agent can observe from the environment
                             Example: {'image': (7, 7, 3), 'direction': 4}
            action_space (int): Number of actions this agent can take
                               Example: 6 for MultiGrid environments
            config (Dict): Learning configuration and hyperparameters
                          Example: {'learning_rate': 3e-4, 'gamma': 0.99, ...}
        
        For Beginners:
        This is like enrolling a student and giving them textbooks (networks),
        study guidelines (hyperparameters), and learning materials.
        """
        # Call parent class constructor to set up basic agent properties
        super().__init__(agent_id, obs_space, action_space, config)
        
        # PPO Core Hyperparameters (these control how the agent learns)
        # Discount factor: how much the agent values future rewards vs immediate rewards
        # gamma = 0.99 means future rewards are worth 99% of immediate rewards
        self.gamma = config.get('gamma', 0.99)
        
        # GAE lambda: controls bias-variance tradeoff in advantage estimation
        # Higher values = less bias but more variance, lower values = more bias but less variance
        self.lambda_gae = config.get('lambda_gae', 0.95)
        
        # PPO clipping parameter: prevents too large policy updates
        # This is PPO's key innovation for stable learning
        self.clip_epsilon = config.get('clip_epsilon', 0.2)  # 20% maximum policy change
        
        # Loss function coefficients: balance different learning objectives
        self.value_loss_coef = config.get('value_loss_coef', 0.5)    # How much to weight value learning
        self.entropy_coef = config.get('entropy_coef', 0.01)         # Encourages exploration
        
        # Gradient clipping: prevents exploding gradients that can destabilize training
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        
        # Training schedule parameters
        self.ppo_epochs = config.get('ppo_epochs', 4)               # How many times to reuse each batch
        self.mini_batch_size = config.get('mini_batch_size', 64)    # Size of training mini-batches
        
        # Neural Networks Setup
        # Actor-Critic architecture: combines policy (actor) and value estimation (critic)
        # This is more efficient than having separate networks
        self.actor_critic = ActorCriticNetwork(
            obs_space, action_space, config, agent_id
        ).to(self.device)
        
        # Optimizer: Adam is standard for neural network training
        # Learning rate controls how big steps to take during learning
        self.optimizer = Adam(
            self.actor_critic.parameters(),      # Parameters to optimize
            lr=config.get('lr', 3e-4),          # Learning rate (3e-4 = 0.0003)
            eps=1e-5                            # Small constant for numerical stability
        )
        
        # Learning rate scheduler: gradually reduces learning rate over time
        # This helps with convergence - start with large steps, end with small steps
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=config.get('lr_decay_steps', 1000),  # Reduce LR every 1000 steps
            gamma=config.get('lr_decay', 0.99)            # Multiply LR by 0.99 each decay
        )
        
        # Experience storage for on-policy learning
        # PPO is "on-policy" meaning it learns from recently collected experiences
        self.reset_memory()  # Initialize empty memory buffers
        
        print(f"Initialized IPPO Agent {agent_id}")
        print(f"Network parameters: {sum(p.numel() for p in self.actor_critic.parameters())}")
        print(f"Learning rate: {config.get('lr', 3e-4)}")
    
    def get_action(self, observation: Dict, training: bool = True) -> Tuple[int, float, float]:
        """
        Select an action given the current observation using the learned policy.
        
        This is the core decision-making method. The agent uses its actor network
        to compute action probabilities, then either samples from this distribution
        (during training) or chooses the most likely action (during evaluation).
        
        Args:
            observation (Dict): Current state observation from environment
                               Example: {'image': grid_state, 'direction': facing_dir}
            training (bool): Whether agent is in training mode
                            True: Sample actions for exploration
                            False: Choose best action deterministically
            
        Returns:
            Tuple[int, float, float]: (action, log_probability, value_estimate)
                action: Integer ID of chosen action
                log_probability: Log probability of chosen action (for PPO updates)
                value_estimate: Critic's estimate of state value
        
        For Beginners:
        This is the agent's "decision-making process". It looks at the current
        situation and decides what to do, while also estimating how good the
        current situation is.
        """
        # Disable gradient computation for action selection (saves memory and computation)
        with torch.no_grad():
            # Convert observation to tensor format for neural network
            obs_tensor = self._process_observation(observation)
            
            # Forward pass through actor-critic network
            # action_logits: unnormalized probabilities for each action
            # value: estimated value of current state
            action_logits, value = self.actor_critic(obs_tensor)
            
            if training:
                # Training mode: sample from policy distribution for exploration
                # Create categorical distribution from action logits
                dist = Categorical(logits=action_logits)
                # Sample an action according to the learned probabilities
                action = dist.sample()
                # Compute log probability of chosen action (needed for PPO loss)
                log_prob = dist.log_prob(action)
            else:
                # Evaluation mode: choose most likely action (greedy/deterministic)
                action = torch.argmax(action_logits, dim=-1)
                # Still compute log probability for consistency
                dist = Categorical(logits=action_logits)
                log_prob = dist.log_prob(action)
            
            # Return action and associated information
            return action.item(), log_prob.item(), value.item()
    
    def _process_observation(self, observation: Dict) -> Dict[str, torch.Tensor]:
        """
        Convert observation dictionary to tensor format for neural networks.
        
        Different environments provide observations in different formats (images,
        vectors, scalars). This method standardizes them into tensor format.
        
        Args:
            observation (Dict): Raw observation from environment
            
        Returns:
            Dict[str, torch.Tensor]: Processed observation tensors
        
        For Beginners:
        This is like translating the environment's "language" into the neural
        network's "language". Each type of information gets converted to numbers.
        """
        processed = {}
        for key, value in observation.items():
            if isinstance(value, np.ndarray):
                # Convert numpy arrays to PyTorch tensors
                processed[key] = torch.tensor(value, dtype=torch.float32).unsqueeze(0).to(self.device)
            else:
                # Convert scalars to single-element tensors
                processed[key] = torch.tensor([value], dtype=torch.float32).to(self.device)
        return processed
    
    def store_transition(self, observation: Dict, action: int, log_prob: float, 
                        value: float, reward: float, terminated: bool, truncated: bool = False):
        """
        Store a single experience transition in the agent's memory.
        
        PPO collects a batch of experiences before updating, so each transition
        is stored until there are enough for a training update.
        
        Args:
            observation (Dict): State where action was taken
            action (int): Action that was taken
            log_prob (float): Log probability of the action under current policy
            value (float): Value estimate of the state
            reward (float): Reward received after taking action
            terminated (bool): Whether episode ended naturally (goal/failure)
            truncated (bool): Whether episode was truncated (time limit)
        
        For Beginners:
        This is like writing in a diary: "In situation X, I did action Y,
        got reward Z, and the episode ended naturally/was cut short/continued."
        
        Note: The distinction between terminated and truncated is crucial for
        correct value function bootstrapping in reinforcement learning.
        """
        self.memory['observations'].append(observation)
        self.memory['actions'].append(action)
        self.memory['log_probs'].append(log_prob)
        self.memory['values'].append(value)
        self.memory['rewards'].append(reward)
        self.memory['terminated'].append(terminated)
        self.memory['truncated'].append(truncated)
        # Keep 'dones' for backward compatibility
        self.memory['dones'].append(terminated or truncated)
    
    def update(self, next_value: float = 0.0) -> Dict[str, float]:
        """
        Update the agent's policy using collected experiences (PPO update).
        
        This implements the core PPO algorithm:
        1. Compute advantages using Generalized Advantage Estimation (GAE)
        2. Perform multiple epochs of policy and value function updates
        3. Use clipping to prevent too-large policy changes
        
        Args:
            next_value (float): Value estimate of next state (for bootstrapping)
                               Used to compute accurate advantage estimates
            
        Returns:
            Dict[str, float]: Training metrics and loss values
                             Example: {'policy_loss': 0.05, 'value_loss': 0.03, ...}
        
        For Beginners:
        This is the "learning phase" where the agent studies its recent experiences
        and updates its strategy to perform better in the future.
        """
        # Check if we have any experiences to learn from
        if len(self.memory['rewards']) == 0:
            return {}  # Nothing to learn from yet
        
        # Convert experience lists to tensors for batch processing
        observations = self.memory['observations']  # Keep as list for now
        actions = torch.tensor(self.memory['actions'], dtype=torch.long).to(self.device)
        old_log_probs = torch.tensor(self.memory['log_probs'], dtype=torch.float32).to(self.device)
        values = torch.tensor(self.memory['values'], dtype=torch.float32).to(self.device)
        rewards = torch.tensor(self.memory['rewards'], dtype=torch.float32).to(self.device)
        
        # CRITICAL: Use terminated for correct value bootstrapping, not done!
        # terminated = True means natural episode end (no bootstrap)
        # terminated = False means continue or time limit (should bootstrap)
        terminated = torch.tensor(self.memory['terminated'], dtype=torch.float32).to(self.device)
        dones = torch.tensor(self.memory['dones'], dtype=torch.float32).to(self.device)  # For compatibility
        
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        # This estimates "how much better was this action compared to average"
        # IMPORTANT: Use terminated (not dones) for correct bootstrapping
        next_values = torch.cat([values[1:], torch.tensor([next_value]).to(self.device)])
        advantages = compute_gae(
            rewards.unsqueeze(1), values.unsqueeze(1), next_values.unsqueeze(1),
            terminated.unsqueeze(1), self.gamma, self.lambda_gae  # Use terminated here!
        ).squeeze(1)
        
        # Returns = advantages + baseline values (what we're trying to predict)
        returns = advantages + values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        
        # Multiple epochs of optimization
        for epoch in range(self.ppo_epochs):
            # Create mini-batches
            batch_size = len(rewards)
            indices = torch.randperm(batch_size)
            
            for start_idx in range(0, batch_size, self.mini_batch_size):
                end_idx = min(start_idx + self.mini_batch_size, batch_size)
                batch_indices = indices[start_idx:end_idx]
                
                # Get batch data
                batch_obs = [observations[i] for i in batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Forward pass
                batch_obs_processed = []
                for obs in batch_obs:
                    batch_obs_processed.append(self._process_observation(obs))
                
                # Stack observations
                stacked_obs = {}
                for key in batch_obs_processed[0].keys():
                    stacked_obs[key] = torch.cat([obs[key] for obs in batch_obs_processed], dim=0)
                
                action_logits, batch_values = self.actor_critic(stacked_obs)
                dist = Categorical(logits=action_logits)
                
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy()
                
                # Policy loss (PPO clip)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(batch_values.squeeze(), batch_returns)
                
                # Entropy loss (for exploration)
                entropy_loss = -entropy.mean()
                
                # Total loss
                total_loss = (policy_loss + 
                             self.value_loss_coef * value_loss + 
                             self.entropy_coef * entropy_loss)
                
                # Optimization step
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
        
        # Update learning rate
        self.lr_scheduler.step()
        
        # Clear memory
        self.reset_memory()
        
        return {
            'policy_loss': total_policy_loss / (self.ppo_epochs * max(1, batch_size // self.mini_batch_size)),
            'value_loss': total_value_loss / (self.ppo_epochs * max(1, batch_size // self.mini_batch_size)),
            'entropy_loss': total_entropy_loss / (self.ppo_epochs * max(1, batch_size // self.mini_batch_size)),
            'mean_reward': rewards.mean().item(),
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
    
    def reset_memory(self):
        """Reset the agent's memory."""
        self.memory = {
            'observations': [],
            'actions': [],
            'log_probs': [],
            'values': [],
            'rewards': [],
            'terminated': [],  # NEW: Natural episode endings
            'truncated': [],   # NEW: Time limit endings
            'dones': []        # Keep for backward compatibility
        }
    
    def save_model(self, path: str):
        """Save the agent's model to disk."""
        checkpoint = {
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'agent_id': self.agent_id,
            'config': self.config
        }
        torch.save(checkpoint, f"{path}_agent_{self.agent_id}.pth")
        print(f"Saved IPPO Agent {self.agent_id} model to {path}_agent_{self.agent_id}.pth")
    
    def load_model(self, path: str):
        """Load the agent's model from disk."""
        checkpoint = torch.load(f"{path}_agent_{self.agent_id}.pth", map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded IPPO Agent {self.agent_id} model from {path}_agent_{self.agent_id}.pth")


class ActorCriticNetwork(nn.Module):
    """
    Improved Actor-Critic network for IPPO with better architecture.
    """
    
    def __init__(self, obs_space: Dict, action_space: int, config: Dict, agent_id: int):
        """
        Initialize the Actor-Critic network.
        
        Args:
            obs_space: Observation space specification
            action_space: Number of available actions
            config: Configuration dictionary
            agent_id: Agent identifier
        """
        super().__init__()
        
        self.obs_space = obs_space
        self.action_space = action_space
        self.config = config
        self.agent_id = agent_id
        
        # Use the existing MultiGridNetwork as base but enhance it
        self.base_network = MultiGridNetwork(
            obs_space, config, action_space, 1, agent_id  # n_agents=1 for individual agent
        )
        
        # Get the output dimension of the base network
        with torch.no_grad():
            sample_obs = self._create_sample_obs()
            base_output = self.base_network(sample_obs)
            base_output_dim = base_output.shape[-1] if len(base_output.shape) > 1 else base_output.shape[0]
        
        # Separate heads for actor and critic
        hidden_dim = config.get('hidden_dim', 128)
        
        # Actor head (policy)
        self.actor_head = nn.Sequential(
            nn.Linear(base_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_space)
        )
        
        # Critic head (value function)
        self.critic_head = nn.Sequential(
            nn.Linear(base_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _create_sample_obs(self) -> Dict:
        """Create a sample observation for network initialization."""
        # This is a placeholder - should match the actual observation format
        sample_obs = {
            'image': torch.zeros((1, 7, 7, 3)),  # Typical multigrid observation
            'direction': torch.tensor([0])
        }
        return sample_obs
    
    def _init_weights(self):
        """Initialize network weights using orthogonal initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Conv2d):
                nn.init.orthogonal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(self, observation: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            observation: Dictionary containing observation data
            
        Returns:
            Tuple of (action_logits, value_estimate)
        """
        # Get shared features from base network
        if hasattr(self.base_network, 'forward'):
            shared_features = self.base_network(observation)
        else:
            # Fallback if base network doesn't have forward method
            shared_features = observation['image'].flatten(start_dim=1)
        
        # Ensure proper shape
        if len(shared_features.shape) == 1:
            shared_features = shared_features.unsqueeze(0)
        
        # Actor and critic outputs
        action_logits = self.actor_head(shared_features)
        value_estimate = self.critic_head(shared_features)
        
        return action_logits, value_estimate


class IPPO(MARLAlgorithm):
    """
    Independent Proximal Policy Optimization (IPPO) algorithm.
    
    Each agent learns independently using PPO while sharing the environment.
    """
    
    def __init__(self, env, config: Dict, device: torch.device):
        """
        Initialize the IPPO algorithm.
        
        Args:
            env: Multi-agent environment
            config: Configuration dictionary
            device: PyTorch device
        """
        super().__init__(env, config, device)
        
        # IPPO specific parameters
        self.rollout_length = config.get('rollout_length', 128)
        self.update_frequency = config.get('update_frequency', self.rollout_length)
        
        print(f"Initialized IPPO with {self.n_agents} agents")
        print(f"Rollout length: {self.rollout_length}")
        print(f"Update frequency: {self.update_frequency}")
    
    def _create_agents(self):
        """Create IPPO agents for each position in the environment."""
        obs_space = self._get_obs_space()
        action_space = len(self.env.actions)
        
        for i in range(self.n_agents):
            agent = IPPOAgent(i, obs_space, action_space, self.config)
            self.agents.append(agent)
            print(f"Created IPPO Agent {i}")
    
    def _get_obs_space(self) -> Dict:
        """Get observation space specification from environment."""
        reset_result = self.env.reset()
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            # New API: (obs, info)
            sample_obs, _ = reset_result
        else:
            # Old API: just obs
            sample_obs = reset_result
            
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
        Collect a rollout of experiences from the environment.
        
        Args:
            env: The environment to collect from
            
        Returns:
            Dictionary containing the collected experiences
        """
        observations = []
        actions = []
        rewards = []
        dones = []
        values = []
        log_probs = []
        
        # Reset environment with new API
        reset_result = env.reset()
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            # New API: (obs, info)
            obs, info = reset_result
        else:
            # Old API: just obs
            obs = reset_result
            info = {}
            
        done = False
        terminated = False
        truncated = False
        step_count = 0
        
        while not done and step_count < self.rollout_length:
            # Get actions from all agents
            agent_actions = []
            agent_log_probs = []
            agent_values = []
            
            for i, agent in enumerate(self.agents):
                # Extract individual agent observation
                if isinstance(obs, dict):
                    agent_obs = {}
                    for key, value in obs.items():
                        if isinstance(value, list):
                            agent_obs[key] = value[i] if i < len(value) else value[0]
                        else:
                            agent_obs[key] = value
                else:
                    agent_obs = obs[i] if isinstance(obs, list) else obs
                
                action, log_prob, value = agent.get_action(agent_obs, training=True)
                agent_actions.append(action)
                agent_log_probs.append(log_prob)
                agent_values.append(value)
            
            # Store current step data
            observations.append(obs)
            actions.append(agent_actions)
            log_probs.append(agent_log_probs)
            values.append(agent_values)
            
            # Take environment step
            step_result = env.step(agent_actions)
            
            # Handle both old and new API formats
            if len(step_result) == 4:
                # Old API: (obs, reward, done, info)
                next_obs, reward, done, info = step_result
                terminated = done
                truncated = False
            elif len(step_result) == 5:
                # New API: (obs, reward, terminated, truncated, info)
                next_obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                raise ValueError(f"Unexpected step return format: {len(step_result)} values")
            
            # Store rewards and episode ending flags
            if isinstance(reward, list):
                rewards.append(reward)
            else:
                rewards.append([reward] * self.n_agents)
            
            dones.append([done] * self.n_agents)
            terminated_flags = [terminated] * self.n_agents
            truncated_flags = [truncated] * self.n_agents
            
            # Store transitions in agent memories
            for i, agent in enumerate(self.agents):
                if isinstance(obs, dict):
                    agent_obs = {}
                    for key, value in obs.items():
                        if isinstance(value, list):
                            agent_obs[key] = value[i] if i < len(value) else value[0]
                        else:
                            agent_obs[key] = value
                else:
                    agent_obs = obs[i] if isinstance(obs, list) else obs
                
                agent_reward = reward[i] if isinstance(reward, list) else reward
                agent.store_transition(
                    agent_obs, agent_actions[i], agent_log_probs[i], 
                    agent_values[i], agent_reward, terminated_flags[i], truncated_flags[i]
                )
            
            obs = next_obs
            step_count += 1
            self.total_steps += 1
        
        # Get final values for bootstrapping
        # CRITICAL: Only bootstrap if episode was truncated (time limit), not terminated (natural end)
        final_values = []
        if not terminated:  # Changed from 'not done' to 'not terminated'
            for i, agent in enumerate(self.agents):
                if isinstance(obs, dict):
                    agent_obs = {}
                    for key, value in obs.items():
                        if isinstance(value, list):
                            agent_obs[key] = value[i] if i < len(value) else value[0]
                        else:
                            agent_obs[key] = value
                else:
                    agent_obs = obs[i] if isinstance(obs, list) else obs
                
                _, _, value = agent.get_action(agent_obs, training=False)
                final_values.append(value)
        else:
            final_values = [0.0] * self.n_agents
        
        return {
            'observations': observations,
            'actions': actions,
            'rewards': rewards,
            'dones': dones,
            'values': values,
            'log_probs': log_probs,
            'final_values': final_values,
            'episode_length': step_count,
            'total_reward': [sum(r[i] for r in rewards) for i in range(self.n_agents)]
        }
    
    def train_step(self, rollout_data: Dict) -> Dict[str, float]:
        """
        Perform one training step using the collected rollout data.
        
        Args:
            rollout_data: Data collected from environment rollouts
            
        Returns:
            Dictionary containing training metrics
        """
        metrics = {}
        
        # Update each agent independently
        for i, agent in enumerate(self.agents):
            agent_metrics = agent.update(rollout_data['final_values'][i])
            
            # Aggregate metrics
            for key, value in agent_metrics.items():
                if key not in metrics:
                    metrics[key] = []
                metrics[key].append(value)
        
        # Average metrics across agents
        averaged_metrics = {}
        for key, values in metrics.items():
            averaged_metrics[f'mean_{key}'] = np.mean(values)
            averaged_metrics[f'std_{key}'] = np.std(values)
        
        # Add episode-level metrics
        averaged_metrics['episode_length'] = rollout_data['episode_length']
        averaged_metrics['total_steps'] = self.total_steps
        averaged_metrics['mean_episode_reward'] = np.mean(rollout_data['total_reward'])
        
        return averaged_metrics
    
    def train_step_vectorized(self, rollout_data: Dict) -> Dict[str, float]:
        """
        Perform one training step using vectorized rollout data.
        
        This method handles data from multiple parallel environments for
        significantly improved training efficiency.
        
        Args:
            rollout_data: Vectorized data from parallel environments
                        Contains: observations, actions, rewards, dones, n_envs
                        
        Returns:
            Dictionary containing training metrics
        """
        n_envs = rollout_data.get('n_envs', 1)
        episode_rewards = rollout_data.get('episode_rewards', [])
        episode_lengths = rollout_data.get('episode_lengths', [])
        
        # Process vectorized data for each agent
        metrics = {}
        
        # Convert vectorized data to format suitable for agent updates
        for i, agent in enumerate(self.agents):
            # Aggregate experiences from all environments for this agent
            agent_experiences = self._aggregate_agent_experiences(rollout_data, i, n_envs)
            
            # Update agent with aggregated experiences
            if agent_experiences['observations']:
                agent_metrics = agent.update(final_value=0.0)  # Vectorized environments auto-reset
                
                # Aggregate metrics
                for key, value in agent_metrics.items():
                    if key not in metrics:
                        metrics[key] = []
                    metrics[key].append(value)
        
        # Average metrics across agents
        averaged_metrics = {}
        for key, values in metrics.items():
            if values:  # Only average if we have values
                averaged_metrics[f'mean_{key}'] = np.mean(values)
                averaged_metrics[f'std_{key}'] = np.std(values)
        
        # Add vectorized episode-level metrics
        averaged_metrics['episode_length'] = np.mean(episode_lengths) if episode_lengths else 0
        averaged_metrics['total_steps'] = self.total_steps
        averaged_metrics['mean_episode_reward'] = np.mean(episode_rewards) if episode_rewards else 0
        averaged_metrics['vectorized_envs'] = n_envs
        averaged_metrics['total_reward_all_envs'] = np.sum(episode_rewards) if episode_rewards else 0
        averaged_metrics['min_episode_reward'] = np.min(episode_rewards) if episode_rewards else 0
        averaged_metrics['max_episode_reward'] = np.max(episode_rewards) if episode_rewards else 0
        averaged_metrics['std_episode_reward'] = np.std(episode_rewards) if episode_rewards else 0
        
        return averaged_metrics
    
    def _aggregate_agent_experiences(self, rollout_data: Dict, agent_idx: int, n_envs: int) -> Dict:
        """
        Aggregate experiences for a specific agent from all vectorized environments.
        
        Args:
            rollout_data: Vectorized rollout data
            agent_idx: Index of the agent to aggregate for
            n_envs: Number of parallel environments
            
        Returns:
            Dictionary containing aggregated experiences for the agent
        """
        observations = rollout_data.get('observations', [])
        actions = rollout_data.get('actions', [])
        rewards = rollout_data.get('rewards', [])
        
        agent_obs = []
        agent_actions = []
        agent_rewards = []
        
        # Extract agent data from all environments and time steps
        for step_obs in observations:
            for env_idx in range(min(len(step_obs), n_envs)):
                if isinstance(step_obs[env_idx], list) and len(step_obs[env_idx]) > agent_idx:
                    # Multi-agent observation
                    agent_obs.append(step_obs[env_idx][agent_idx])
                elif not isinstance(step_obs[env_idx], list):
                    # Single agent observation
                    agent_obs.append(step_obs[env_idx])
        
        for step_actions in actions:
            for env_idx in range(min(len(step_actions), n_envs)):
                if isinstance(step_actions[env_idx], list) and len(step_actions[env_idx]) > agent_idx:
                    agent_actions.append(step_actions[env_idx][agent_idx])
                elif not isinstance(step_actions[env_idx], list):
                    agent_actions.append(step_actions[env_idx])
        
        for step_rewards in rewards:
            for env_idx in range(min(len(step_rewards), n_envs)):
                if isinstance(step_rewards[env_idx], list) and len(step_rewards[env_idx]) > agent_idx:
                    agent_rewards.append(step_rewards[env_idx][agent_idx])
                elif not isinstance(step_rewards[env_idx], list):
                    agent_rewards.append(step_rewards[env_idx])
        
        return {
            'observations': agent_obs,
            'actions': agent_actions,
            'rewards': agent_rewards
        }
    
    def get_actions_batch(self, obs_batch, training: bool = True):
        """
        Get actions for a batch of observations efficiently.
        
        Args:
            obs_batch: Batch of observations
            training: Whether in training mode
            
        Returns:
            Batch of actions
        """
        if not isinstance(obs_batch, list):
            obs_batch = [obs_batch]
        
        actions = []
        for obs in obs_batch:
            agent_actions = []
            for i, agent in enumerate(self.agents):
                agent_obs = self._extract_agent_obs(obs, i)
                action, _, _ = agent.get_action(agent_obs, training=training)
                agent_actions.append(action)
            actions.append(agent_actions)
        
        return actions[0] if len(actions) == 1 else actions
    
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
            # Reset environment with new API
            reset_result = env.reset()
            if isinstance(reset_result, tuple) and len(reset_result) == 2:
                # New API: (obs, info)
                obs, info = reset_result
            else:
                # Old API: just obs
                obs = reset_result
                info = {}
                
            done = False
            terminated = False
            truncated = False
            episode_reward = [0.0] * self.n_agents
            episode_length = 0
            
            while not done:
                # Get actions from all agents (greedy)
                agent_actions = []
                
                for i, agent in enumerate(self.agents):
                    if isinstance(obs, dict):
                        agent_obs = {}
                        for key, value in obs.items():
                            if isinstance(value, list):
                                agent_obs[key] = value[i] if i < len(value) else value[0]
                            else:
                                agent_obs[key] = value
                    else:
                        agent_obs = obs[i] if isinstance(obs, list) else obs
                    
                    action, _, _ = agent.get_action(agent_obs, training=False)
                    agent_actions.append(action)
                
                # Take environment step
                step_result = env.step(agent_actions)
                
                # Handle both old and new API formats
                if len(step_result) == 4:
                    # Old API: (obs, reward, done, info)
                    obs, reward, done, info = step_result
                    terminated = done
                    truncated = False
                elif len(step_result) == 5:
                    # New API: (obs, reward, terminated, truncated, info)
                    obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:
                    raise ValueError(f"Unexpected step return format: {len(step_result)} values")
                
                # Accumulate rewards
                if isinstance(reward, list):
                    for i in range(self.n_agents):
                        episode_reward[i] += reward[i]
                else:
                    for i in range(self.n_agents):
                        episode_reward[i] += reward
                
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
