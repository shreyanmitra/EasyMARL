"""
Multi-Agent Proximal Policy Optimization (MAPPO) algorithm.

MAPPO uses centralized value functions during training while maintaining
decentralized policies for execution. This implementation includes parameter
sharing options and advanced multi-agent features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim import Adam
import numpy as np
from typing import Dict, List, Tuple, Any

from .base import MARLAgent, MARLAlgorithm, compute_gae
from networks.multigrid_network import MultiGridNetwork


class MAPPOAgent(MARLAgent):
    """
    MAPPO agent with centralized value function and decentralized policy.
    """
    
    def __init__(self, agent_id: int, obs_space: Dict, action_space: int, 
                 global_state_dim: int, n_agents: int, config: Dict):
        """
        Initialize the MAPPO agent.
        
        Args:
            agent_id: Unique identifier for this agent
            obs_space: Individual observation space
            action_space: Number of available actions
            global_state_dim: Dimension of global state for centralized critic
            n_agents: Total number of agents
            config: Configuration dictionary
        """
        super().__init__(agent_id, obs_space, action_space, config)
        
        self.global_state_dim = global_state_dim
        self.n_agents = n_agents
        
        # PPO hyperparameters
        self.gamma = config.get('gamma', 0.99)
        self.lambda_gae = config.get('lambda_gae', 0.95)
        self.clip_epsilon = config.get('clip_epsilon', 0.2)
        self.value_loss_coef = config.get('value_loss_coef', 0.5)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        
        # Networks
        self.actor = MAPPOActor(obs_space, action_space, config).to(self.device)
        
        # Centralized critic takes global state
        self.critic = MAPPOCritic(global_state_dim, config).to(self.device)
        
        # Optimizers
        self.actor_optimizer = Adam(
            self.actor.parameters(), 
            lr=config.get('lr_actor', 3e-4),
            eps=1e-5
        )
        self.critic_optimizer = Adam(
            self.critic.parameters(), 
            lr=config.get('lr_critic', 1e-3),
            eps=1e-5
        )
        
        print(f"Initialized MAPPO Agent {agent_id}")
        print(f"Actor parameters: {sum(p.numel() for p in self.actor.parameters())}")
        print(f"Critic parameters: {sum(p.numel() for p in self.critic.parameters())}")
    
    def get_action(self, observation: Dict, global_state: np.ndarray, 
                   training: bool = True) -> Tuple[int, float, float]:
        """
        Select an action and get value estimate.
        
        Args:
            observation: Individual agent observation
            global_state: Global state for value estimation
            training: Whether in training mode
            
        Returns:
            Tuple of (action, log_probability, value_estimate)
        """
        with torch.no_grad():
            obs_tensor = self._process_observation(observation)
            state_tensor = torch.tensor(global_state, dtype=torch.float32).unsqueeze(0).to(self.device)
            
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
            
            # Get value estimate from centralized critic
            value = self.critic(state_tensor)
            
            return action.item(), log_prob.item(), value.item()
    
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
    
    def store_transition(self, observation: Dict, global_state: np.ndarray, 
                        action: int, log_prob: float, value: float, 
                        reward: float, done: bool):
        """Store a transition in memory."""
        self.memory['observations'].append(observation)
        self.memory['global_states'].append(global_state)
        self.memory['actions'].append(action)
        self.memory['log_probs'].append(log_prob)
        self.memory['values'].append(value)
        self.memory['rewards'].append(reward)
        self.memory['dones'].append(done)
    
    def update_actor(self, batch_obs: List[Dict], batch_actions: torch.Tensor,
                     batch_old_log_probs: torch.Tensor, batch_advantages: torch.Tensor) -> float:
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
        
        new_log_probs = dist.log_prob(batch_actions)
        entropy = dist.entropy()
        
        # PPO policy loss
        ratio = torch.exp(new_log_probs - batch_old_log_probs)
        surr1 = ratio * batch_advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
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
    
    def update_critic(self, batch_global_states: torch.Tensor, 
                     batch_returns: torch.Tensor) -> float:
        """Update the centralized critic."""
        # Forward pass
        values = self.critic(batch_global_states).squeeze()
        
        # Value loss
        value_loss = F.mse_loss(values, batch_returns)
        
        # Update critic
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()
        
        return value_loss.item()
    
    def reset_memory(self):
        """Reset agent memory."""
        self.memory = {
            'observations': [],
            'global_states': [],
            'actions': [],
            'log_probs': [],
            'values': [],
            'rewards': [],
            'dones': []
        }
    
    def save_model(self, path: str):
        """Save the agent's model."""
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'agent_id': self.agent_id,
            'config': self.config
        }
        torch.save(checkpoint, f"{path}_mappo_agent_{self.agent_id}.pth")
        print(f"Saved MAPPO Agent {self.agent_id} model to {path}_mappo_agent_{self.agent_id}.pth")
    
    def load_model(self, path: str):
        """Load the agent's model."""
        checkpoint = torch.load(f"{path}_mappo_agent_{self.agent_id}.pth", map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
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
