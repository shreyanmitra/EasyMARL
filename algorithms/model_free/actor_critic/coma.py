"""
(C) Shreyan Mitra, based on starter code by Natasha Jaques

Counterfactual Multi-Agent Policy Gradients (COMA) Algorithm

COMA is a sophisticated multi-agent policy gradient method that solves the
multi-agent credit assignment problem using counterfactual reasoning.

Key Innovation - Counterfactual Advantage:
Instead of asking "How good was this action?", COMA asks:
"How much better was this action compared to the average action this agent could have taken?"

How COMA Works:
1. Centralized Critic: Learns Q(s, u₁, u₂, ..., uₙ) - joint action-value function
2. Counterfactual Baseline: For agent i, compute Q(s, u₋ᵢ, uᵢ) vs Q(s, u₋ᵢ, avg(uᵢ))
3. Individual Advantage: Advantage = Q(actual) - Q(counterfactual baseline)
4. Policy Gradient: Update each agent's policy using its individual advantage

When to Use COMA:
✅ Mixed cooperative-competitive environments
✅ Need for sophisticated credit assignment
✅ Discrete action spaces (original version)
✅ When you want to understand each agent's contribution
✅ Environments where individual agent impact is unclear

Key Advantages:
✅ Addresses multi-agent credit assignment problem
✅ Theoretically grounded advantage estimation
✅ Works well with heterogeneous agents
✅ Provides interpretable individual contributions

Limitations:
❌ High computational cost (needs joint action space)
❌ Originally designed for discrete actions
❌ Complex to implement correctly
❌ Requires careful tuning

Comparison with Other Algorithms:
- vs MAPPO: More sophisticated credit assignment but higher computational cost
- vs QMIX: Policy gradients vs value-based, better for continuous control
- vs MADDPG: Better credit assignment but more complex

For MARL Beginners:
COMA is advanced! Start with IPPO or MAPPO first. COMA addresses the question:
"In a team setting, how do we know which team member contributed to success?"
It's like having a coach who can evaluate each player's impact by imagining
what would have happened if they had played differently.

Paper: "Counterfactual Multi-Agent Policy Gradients" (2018)
Use Cases: StarCraft II, Capture the Flag, complex coordination tasks
"""

# Import necessary libraries for deep learning and multi-agent systems
import torch                    # PyTorch for neural networks
import torch.nn as nn           # Neural network modules
import torch.nn.functional as F # Activation functions and utilities
from torch.distributions import Categorical  # For sampling from probability distributions
from torch.optim import Adam    # Adam optimizer for gradient-based learning
import numpy as np              # Numerical computations
from typing import Dict, List, Tuple, Any  # Type hints for code clarity

# Import base classes from our MARL framework
from .base import MARLAgent, MARLAlgorithm


class COMAAgent(MARLAgent):
    """
    Counterfactual Multi-Agent Policy Gradients Agent.
    
    This agent implements COMA's decentralized actor with sophisticated
    credit assignment through counterfactual reasoning. Each agent has
    its own policy but benefits from a shared centralized critic that
    can perform counterfactual analysis.
    
    Key Components:
    1. Decentralized Actor: π(action|local_observation) - only uses local info
    2. Centralized Critic: Q(global_state, joint_actions) - sees everything
    3. Counterfactual Baseline: Estimates what would happen with "average" actions
    4. Individual Advantage: Measures each agent's specific contribution
    
    Architecture Insight:
    - Actor Network: Input = local observation → Output = action probabilities
    - Critic Network: Input = (global_state, all_actions) → Output = Q-value
    - Counterfactual: Critic evaluates current actions vs baseline actions
    
    For MARL Beginners:
    Think of this as an agent that plays its own game (decentralized) but
    has access to a "super coach" who can tell it exactly how much its
    individual actions contributed to the team's success or failure.
    """
    
    def __init__(self, agent_id: int, obs_space: Dict, action_space: int, config: Dict):
        """
        Initialize the COMA agent with decentralized actor network.
        
        Args:
            agent_id (int): Unique identifier for this agent
            obs_space (Dict): Local observation space for this agent
                             Example: {'image': (7, 7, 3), 'direction': 4}
            action_space (int): Number of actions this agent can take
                               Example: 6 for MultiGrid environments
            config (Dict): Configuration containing hyperparameters
                          Example: {'gamma': 0.99, 'lr_actor': 3e-4, ...}
        
        For Beginners:
        This sets up an agent that can act independently but will receive
        sophisticated feedback about its individual contributions during training.
        """
        # Call parent class constructor to set up basic agent properties
        super().__init__(agent_id, obs_space, action_space, config)
        
        # COMA Core Hyperparameters
        # Discount factor: how much the agent values future rewards
        self.gamma = config.get('gamma', 0.99)
        
        # GAE lambda: controls bias-variance tradeoff in advantage estimation
        self.lambda_gae = config.get('lambda_gae', 0.95)
        
        # Entropy coefficient: encourages exploration by rewarding diverse actions
        self.entropy_coef = config.get('entropy_coef', 0.01)
        
        # Gradient clipping: prevents exploding gradients
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        
        # Actor Network Setup (Decentralized)
        # This network only sees local observations, maintaining decentralized execution
        self.actor = COMAActorNetwork(obs_space, action_space, config).to(self.device)
        
        # Actor Optimizer for Learning
        self.actor_optimizer = Adam(
            self.actor.parameters(), 
            lr=config.get('lr_actor', 3e-4)  # Learning rate for policy updates
        )
        
        print(f"Initialized COMA Agent {agent_id}")
        print(f"Actor parameters: {sum(p.numel() for p in self.actor.parameters())}")
    
    def get_action(self, observation: Dict, training: bool = True) -> Tuple[int, float]:
        """
        Select an action given the current observation.
        
        Args:
            observation: Individual agent observation
            training: Whether in training mode
            
        Returns:
            Tuple of (action, log_probability)
        """
        with torch.no_grad():
            obs_tensor = self._process_observation(observation)
            
            # Get action from policy
            action_logits = self.actor(obs_tensor)
            
            if training:
                dist = Categorical(logits=action_logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)
            else:
                action = torch.argmax(action_logits, dim=-1)
                dist = Categorical(logits=action_logits)
                log_prob = dist.log_prob(action)
            
            return action.item(), log_prob.item()
    
    def get_action_probs(self, observation: Dict) -> torch.Tensor:
        """Get action probabilities for the given observation."""
        obs_tensor = self._process_observation(observation)
        action_logits = self.actor(obs_tensor)
        return F.softmax(action_logits, dim=-1)
    
    def get_action_logits(self, observation: Dict) -> torch.Tensor:
        """Get raw action logits for the given observation."""
        obs_tensor = self._process_observation(observation)
        return self.actor(obs_tensor)
    
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
    
    def update_actor(self, batch_obs: List[Dict], batch_actions: torch.Tensor,
                     batch_advantages: torch.Tensor) -> float:
        """Update the actor network."""
        # Process observations
        obs_tensors = []
        for obs in batch_obs:
            obs_tensor = self._process_observation(obs)
            obs_tensors.append(obs_tensor)
        
        stacked_obs = torch.cat(obs_tensors, dim=0)
        
        # Forward pass
        action_logits = self.actor(stacked_obs)
        dist = Categorical(logits=action_logits)
        
        log_probs = dist.log_prob(batch_actions)
        entropy = dist.entropy()
        
        # Policy loss with counterfactual advantages
        policy_loss = -(log_probs * batch_advantages).mean()
        
        # Entropy bonus
        entropy_loss = -entropy.mean()
        
        # Total actor loss
        actor_loss = policy_loss + self.entropy_coef * entropy_loss
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()
        
        return actor_loss.item()
    
    def reset_memory(self):
        """Reset agent memory."""
        self.memory = {
            'observations': [],
            'actions': [],
            'log_probs': [],
            'rewards': [],
            'dones': []
        }
    
    def save_model(self, path: str):
        """Save the agent's model."""
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'agent_id': self.agent_id,
            'config': self.config
        }
        torch.save(checkpoint, f"{path}_coma_agent_{self.agent_id}.pth")
        print(f"Saved COMA Agent {self.agent_id} model to {path}_coma_agent_{self.agent_id}.pth")
    
    def load_model(self, path: str):
        """Load the agent's model."""
        checkpoint = torch.load(f"{path}_coma_agent_{self.agent_id}.pth", map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        print(f"Loaded COMA Agent {self.agent_id} model from {path}_coma_agent_{self.agent_id}.pth")


class COMAActorNetwork(nn.Module):
    """Decentralized actor network for COMA agent."""
    
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


class COMACriticNetwork(nn.Module):
    """Centralized critic network for COMA with counterfactual reasoning."""
    
    def __init__(self, global_state_dim: int, n_agents: int, action_space: int, config: Dict):
        super().__init__()
        
        self.n_agents = n_agents
        self.action_space = action_space
        
        hidden_dim = config.get('hidden_dim', 128)
        
        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(global_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Action encoders for other agents
        self.action_encoder = nn.Sequential(
            nn.Linear((n_agents - 1) * action_space, hidden_dim),
            nn.ReLU()
        )
        
        # Q-value head for each agent
        self.q_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim + hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_space)
            ) for _ in range(n_agents)
        ])
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, global_state: torch.Tensor, actions: torch.Tensor, 
               agent_id: int) -> torch.Tensor:
        """
        Forward pass through critic for specific agent.
        
        Args:
            global_state: Global state [batch_size, state_dim]
            actions: Actions of all agents [batch_size, n_agents, action_space] (one-hot)
            agent_id: ID of the agent to get Q-values for
            
        Returns:
            Q-values for all actions of the specified agent [batch_size, action_space]
        """
        batch_size = global_state.shape[0]
        
        # Encode state
        state_features = self.state_encoder(global_state)
        
        # Get actions of other agents (exclude target agent)
        other_actions = []
        for i in range(self.n_agents):
            if i != agent_id:
                other_actions.append(actions[:, i])
        
        other_actions = torch.cat(other_actions, dim=-1)  # [batch_size, (n_agents-1) * action_space]
        action_features = self.action_encoder(other_actions)
        
        # Combine features
        combined_features = torch.cat([state_features, action_features], dim=-1)
        
        # Get Q-values for the target agent
        q_values = self.q_heads[agent_id](combined_features)
        
        return q_values


class COMA(MARLAlgorithm):
    """
    Counterfactual Multi-Agent Policy Gradients (COMA) algorithm.
    
    Uses centralized critic with counterfactual reasoning and decentralized actors.
    """
    
    def __init__(self, env, config: Dict, device: torch.device):
        """
        Initialize COMA algorithm.
        
        Args:
            env: Multi-agent environment
            config: Configuration dictionary
            device: PyTorch device
        """
        super().__init__(env, config, device)
        
        # COMA specific parameters
        self.rollout_length = config.get('rollout_length', 128)
        self.critic_epochs = config.get('critic_epochs', 5)
        self.actor_epochs = config.get('actor_epochs', 1)
        
        # Global state dimension
        self.global_state_dim = self._get_global_state_dim()
        
        # Create centralized critic
        self.critic = COMACriticNetwork(
            self.global_state_dim, self.n_agents, len(self.env.actions), config
        ).to(device)
        
        self.critic_optimizer = Adam(
            self.critic.parameters(), 
            lr=config.get('lr_critic', 1e-3)
        )
        
        print(f"Initialized COMA")
        print(f"Rollout length: {self.rollout_length}")
        print(f"Global state dim: {self.global_state_dim}")
        print(f"Critic parameters: {sum(p.numel() for p in self.critic.parameters())}")
    
    def _create_agents(self):
        """Create COMA agents."""
        obs_space = self._get_obs_space()
        action_space = len(self.env.actions)
        
        for i in range(self.n_agents):
            agent = COMAAgent(i, obs_space, action_space, self.config)
            self.agents.append(agent)
    
    def _get_obs_space(self) -> Dict:
        """Get observation space specification."""
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
        """Collect rollout for COMA training."""
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
        step_count = 0
        
        # Initialize data storage
        episode_data = {
            'observations': [[] for _ in range(self.n_agents)],
            'global_states': [],
            'actions': [[] for _ in range(self.n_agents)],
            'action_probs': [[] for _ in range(self.n_agents)],
            'rewards': [[] for _ in range(self.n_agents)],
            'dones': [[] for _ in range(self.n_agents)],
            'terminated': [[] for _ in range(self.n_agents)],
            'truncated': [[] for _ in range(self.n_agents)]
        }
        
        while not done and step_count < self.rollout_length:
            # Get global state
            global_state = self._get_global_state(obs)
            episode_data['global_states'].append(global_state)
            
            # Get actions from all agents
            actions = []
            action_probs = []
            
            for i, agent in enumerate(self.agents):
                agent_obs = self._extract_agent_obs(obs, i)
                action, log_prob = agent.get_action(agent_obs, training=True)
                
                # Get action probabilities for counterfactual reasoning
                probs = agent.get_action_probs(agent_obs)
                
                actions.append(action)
                action_probs.append(probs)
                
                # Store in episode data
                episode_data['observations'][i].append(agent_obs)
                episode_data['actions'][i].append(action)
                episode_data['action_probs'][i].append(probs)
            
            # Take environment step
            step_result = env.step(actions)
            
            # Handle both old and new API formats
            if len(step_result) == 4:
                # Old API: (obs, reward, done, info)
                next_obs, rewards, done, info = step_result
                terminated = done
                truncated = False
            elif len(step_result) == 5:
                # New API: (obs, reward, terminated, truncated, info)
                next_obs, rewards, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                raise ValueError(f"Unexpected step return format: {len(step_result)} values")
            
            # Store rewards, terminated, and truncated flags
            if isinstance(rewards, list):
                for i in range(self.n_agents):
                    episode_data['rewards'][i].append(rewards[i])
                    episode_data['dones'][i].append(done)
                    episode_data['terminated'][i].append(terminated)
                    episode_data['truncated'][i].append(truncated)
            else:
                for i in range(self.n_agents):
                    episode_data['rewards'][i].append(rewards)
                    episode_data['dones'][i].append(done)
                    episode_data['terminated'][i].append(terminated)
                    episode_data['truncated'][i].append(truncated)
            
            obs = next_obs
            step_count += 1
            self.total_steps += 1
        
        return {
            'episode_data': episode_data,
            'episode_length': step_count,
            'total_reward': [sum(episode_data['rewards'][i]) for i in range(self.n_agents)]
        }
    
    def train_step(self, rollout_data: Dict) -> Dict[str, float]:
        """Perform COMA training step."""
        episode_data = rollout_data['episode_data']
        
        # Compute counterfactual advantages
        advantages = self._compute_counterfactual_advantages(episode_data)
        
        # Update critic
        critic_loss = self._update_critic(episode_data)
        
        # Update actors
        actor_losses = []
        for i, agent in enumerate(self.agents):
            actor_loss = self._update_agent_actor(agent, i, episode_data, advantages[i])
            actor_losses.append(actor_loss)
        
        return {
            'critic_loss': critic_loss,
            'mean_actor_loss': np.mean(actor_losses),
            'episode_length': rollout_data['episode_length'],
            'mean_episode_reward': np.mean(rollout_data['total_reward']),
            'total_steps': self.total_steps
        }
    
    def _compute_counterfactual_advantages(self, episode_data: Dict) -> List[torch.Tensor]:
        """Compute counterfactual advantages for each agent."""
        advantages = []
        
        for agent_id in range(self.n_agents):
            agent_advantages = []
            
            for t in range(len(episode_data['global_states'])):
                global_state = torch.tensor(
                    episode_data['global_states'][t], dtype=torch.float32
                ).unsqueeze(0).to(self.device)
                
                # Get one-hot encoded actions
                all_actions = []
                for j in range(self.n_agents):
                    action = episode_data['actions'][j][t]
                    action_onehot = torch.zeros(len(self.env.actions))
                    action_onehot[action] = 1.0
                    all_actions.append(action_onehot)
                
                all_actions = torch.stack(all_actions).unsqueeze(0).to(self.device)  # [1, n_agents, action_space]
                
                # Get Q-values for this agent
                q_values = self.critic(global_state, all_actions, agent_id)  # [1, action_space]
                
                # Compute counterfactual baseline
                baseline = 0.0
                action_probs = episode_data['action_probs'][agent_id][t].to(self.device)
                
                for a in range(len(self.env.actions)):
                    # Create counterfactual action
                    cf_actions = all_actions.clone()
                    cf_action_onehot = torch.zeros(len(self.env.actions)).to(self.device)
                    cf_action_onehot[a] = 1.0
                    cf_actions[0, agent_id] = cf_action_onehot
                    
                    cf_q_values = self.critic(global_state, cf_actions, agent_id)
                    baseline += action_probs[a] * cf_q_values[0, a]
                
                # Advantage = Q(s,u) - baseline
                taken_action = episode_data['actions'][agent_id][t]
                advantage = q_values[0, taken_action] - baseline
                agent_advantages.append(advantage)
            
            advantages.append(torch.stack(agent_advantages))
        
        return advantages
    
    def _update_critic(self, episode_data: Dict) -> float:
        """Update the centralized critic."""
        total_loss = 0.0
        
        for epoch in range(self.critic_epochs):
            for t in range(len(episode_data['global_states']) - 1):
                global_state = torch.tensor(
                    episode_data['global_states'][t], dtype=torch.float32
                ).unsqueeze(0).to(self.device)
                
                next_global_state = torch.tensor(
                    episode_data['global_states'][t + 1], dtype=torch.float32
                ).unsqueeze(0).to(self.device)
                
                # Get actions at time t
                all_actions = []
                next_all_actions = []
                
                for j in range(self.n_agents):
                    action = episode_data['actions'][j][t]
                    action_onehot = torch.zeros(len(self.env.actions))
                    action_onehot[action] = 1.0
                    all_actions.append(action_onehot)
                    
                    next_action = episode_data['actions'][j][t + 1]
                    next_action_onehot = torch.zeros(len(self.env.actions))
                    next_action_onehot[next_action] = 1.0
                    next_all_actions.append(next_action_onehot)
                
                all_actions = torch.stack(all_actions).unsqueeze(0).to(self.device)
                next_all_actions = torch.stack(next_all_actions).unsqueeze(0).to(self.device)
                
                # Update critic for each agent
                for agent_id in range(self.n_agents):
                    current_q = self.critic(global_state, all_actions, agent_id)
                    taken_action = episode_data['actions'][agent_id][t]
                    current_q_value = current_q[0, taken_action]
                    
                    with torch.no_grad():
                        next_q = self.critic(next_global_state, next_all_actions, agent_id)
                        next_taken_action = episode_data['actions'][agent_id][t + 1]
                        next_q_value = next_q[0, next_taken_action]
                        
                        reward = episode_data['rewards'][agent_id][t]
                        done = episode_data['dones'][agent_id][t]
                        target = reward + self.gamma * next_q_value * (1 - done)
                    
                    loss = F.mse_loss(current_q_value, target)
                    
                    self.critic_optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                    self.critic_optimizer.step()
                    
                    total_loss += loss.item()
        
        return total_loss / (self.critic_epochs * len(episode_data['global_states']) * self.n_agents)
    
    def _update_agent_actor(self, agent: COMAAgent, agent_id: int, 
                           episode_data: Dict, advantages: torch.Tensor) -> float:
        """Update a specific agent's actor."""
        # Prepare batch data
        batch_obs = episode_data['observations'][agent_id]
        batch_actions = torch.tensor(
            episode_data['actions'][agent_id], dtype=torch.long
        ).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Update actor
        actor_loss = agent.update_actor(batch_obs, batch_actions, advantages)
        
        return actor_loss
