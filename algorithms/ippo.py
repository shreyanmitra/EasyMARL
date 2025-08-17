"""
Independent Proximal Policy Optimization (IPPO) algorithm for multi-agent environments.

This implementation provides a robust and efficient IPPO algorithm with proper
batching, GAE computation, and modular design for multi-agent scenarios.
"""

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Categorical
    from torch.optim import Adam
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Mock classes for structure testing
    class torch:
        class nn:
            class Module: pass
        class optim:
            class Adam: pass

import numpy as np
from typing import Dict, List, Tuple, Any

try:
    import wandb
except ImportError:
    wandb = None

from .base import MARLAgent, MARLAlgorithm, compute_gae, compute_returns

if TORCH_AVAILABLE:
    from networks.multigrid_network import MultiGridNetwork


class IPPOAgent(MARLAgent):
    """
    Independent PPO agent implementation with proper GAE and modern features.
    """
    
    def __init__(self, agent_id: int, obs_space: Dict, action_space: int, config: Dict):
        """
        Initialize the IPPO agent.
        
        Args:
            agent_id: Unique identifier for this agent
            obs_space: Observation space specification  
            action_space: Number of available actions
            config: Configuration dictionary containing hyperparameters
        """
        super().__init__(agent_id, obs_space, action_space, config)
        
        # PPO hyperparameters
        self.gamma = config.get('gamma', 0.99)
        self.lambda_gae = config.get('lambda_gae', 0.95)
        self.clip_epsilon = config.get('clip_epsilon', 0.2)
        self.value_loss_coef = config.get('value_loss_coef', 0.5)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        self.ppo_epochs = config.get('ppo_epochs', 4)
        self.mini_batch_size = config.get('mini_batch_size', 64)
        
        # Networks
        self.actor_critic = ActorCriticNetwork(
            obs_space, action_space, config, agent_id
        ).to(self.device)
        
        # Optimizer
        self.optimizer = Adam(
            self.actor_critic.parameters(), 
            lr=config.get('lr', 3e-4),
            eps=1e-5
        )
        
        # Learning rate scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=config.get('lr_decay_steps', 1000),
            gamma=config.get('lr_decay', 0.99)
        )
        
        print(f"Initialized IPPO Agent {agent_id} with {sum(p.numel() for p in self.actor_critic.parameters())} parameters")
    
    def get_action(self, observation: Dict, training: bool = True) -> Tuple[int, float, float]:
        """
        Select an action given the current observation.
        
        Args:
            observation: Current observation from the environment
            training: Whether the agent is in training mode
            
        Returns:
            Tuple of (action, log_probability, value_estimate)
        """
        with torch.no_grad():
            obs_tensor = self._process_observation(observation)
            action_logits, value = self.actor_critic(obs_tensor)
            
            if training:
                # Sample from policy distribution
                dist = Categorical(logits=action_logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)
            else:
                # Take greedy action during evaluation
                action = torch.argmax(action_logits, dim=-1)
                dist = Categorical(logits=action_logits)
                log_prob = dist.log_prob(action)
            
            return action.item(), log_prob.item(), value.item()
    
    def _process_observation(self, observation: Dict) -> Dict[str, torch.Tensor]:
        """Convert observation to tensor format."""
        processed = {}
        for key, value in observation.items():
            if isinstance(value, np.ndarray):
                processed[key] = torch.tensor(value, dtype=torch.float32).unsqueeze(0).to(self.device)
            else:
                processed[key] = torch.tensor([value], dtype=torch.float32).to(self.device)
        return processed
    
    def store_transition(self, observation: Dict, action: int, log_prob: float, 
                        value: float, reward: float, done: bool):
        """Store a transition in the agent's memory."""
        self.memory['observations'].append(observation)
        self.memory['actions'].append(action)
        self.memory['log_probs'].append(log_prob)
        self.memory['values'].append(value)
        self.memory['rewards'].append(reward)
        self.memory['dones'].append(done)
    
    def update(self, next_value: float = 0.0) -> Dict[str, float]:
        """
        Update the agent's policy using collected experiences.
        
        Args:
            next_value: Value estimate of the next state (for bootstrapping)
            
        Returns:
            Dictionary containing loss values and metrics
        """
        if len(self.memory['rewards']) == 0:
            return {}
        
        # Convert lists to tensors
        observations = self.memory['observations']
        actions = torch.tensor(self.memory['actions'], dtype=torch.long).to(self.device)
        old_log_probs = torch.tensor(self.memory['log_probs'], dtype=torch.float32).to(self.device)
        values = torch.tensor(self.memory['values'], dtype=torch.float32).to(self.device)
        rewards = torch.tensor(self.memory['rewards'], dtype=torch.float32).to(self.device)
        dones = torch.tensor(self.memory['dones'], dtype=torch.float32).to(self.device)
        
        # Compute advantages and returns using GAE
        next_values = torch.cat([values[1:], torch.tensor([next_value]).to(self.device)])
        advantages = compute_gae(
            rewards.unsqueeze(1), values.unsqueeze(1), next_values.unsqueeze(1),
            dones.unsqueeze(1), self.gamma, self.lambda_gae
        ).squeeze(1)
        
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
            'dones': []
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
        
        obs = env.reset()
        done = False
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
            next_obs, reward, done, info = env.step(agent_actions)
            
            # Store rewards and done flags
            if isinstance(reward, list):
                rewards.append(reward)
            else:
                rewards.append([reward] * self.n_agents)
            
            dones.append([done] * self.n_agents)
            
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
                    agent_values[i], agent_reward, done
                )
            
            obs = next_obs
            step_count += 1
            self.total_steps += 1
        
        # Get final values for bootstrapping
        final_values = []
        if not done:
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
                obs, reward, done, _ = env.step(agent_actions)
                
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
