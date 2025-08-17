"""
Counterfactual Multi-Agent Policy Gradients with Communication (COMA-Comm).

Extends COMA with explicit communication channels to improve
credit assignment and coordination in multi-agent settings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from typing import Dict, List, Tuple, Any
from collections import deque
import random

from .base import MARLAgent, MARLAlgorithm


class COMACommAgent(MARLAgent):
    """
    COMA agent with communication capabilities.
    """
    
    def __init__(self, agent_id: int, obs_space: Dict, action_space: int, config: Dict):
        """
        Initialize the COMA-Comm agent.
        
        Args:
            agent_id: Unique identifier for this agent
            obs_space: Individual observation space
            action_space: Number of available actions
            config: Configuration dictionary
        """
        super().__init__(agent_id, obs_space, action_space, config)
        
        # COMA-Comm hyperparameters
        self.gamma = config.get('gamma', 0.99)
        self.communication_dim = config.get('communication_dim', 32)
        self.td_lambda = config.get('td_lambda', 0.8)
        
        # Actor network (policy) with communication
        self.actor = COMACommActor(obs_space, action_space, self.communication_dim, config).to(self.device)
        
        # Actor optimizer
        self.actor_optimizer = Adam(self.actor.parameters(), lr=config.get('actor_lr', 1e-4))
        
        # Communication state
        self.last_communication = torch.zeros(self.communication_dim).to(self.device)
        self.communication_history = deque(maxlen=config.get('comm_history_length', 10))
        
        # Episode data for TD(λ) computation
        self.episode_data = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'messages': [],
            'log_probs': [],
            'values': []
        }
        
        print(f"Initialized COMA-Comm Agent {agent_id}")
        print(f"Actor parameters: {sum(p.numel() for p in self.actor.parameters())}")
        print(f"Communication dimension: {self.communication_dim}")
    
    def get_action(self, observation: Dict, received_messages: List[torch.Tensor] = None, 
                   training: bool = True) -> Tuple[int, torch.Tensor, float, torch.Tensor]:
        """
        Select an action and generate communication message.
        
        Args:
            observation: Agent observation
            received_messages: Messages from other agents
            training: Whether in training mode
            
        Returns:
            Tuple of (action, communication_message, log_prob, action_probs)
        """
        obs_tensor = self._process_observation(observation)
        
        # Process received messages
        if received_messages:
            aggregated_message = self._aggregate_messages(received_messages)
        else:
            aggregated_message = torch.zeros(self.communication_dim).to(self.device)
        
        # Get action probabilities and communication from actor
        action_probs, communication_message = self.actor(obs_tensor, aggregated_message.unsqueeze(0))
        
        # Sample action
        action_dist = torch.distributions.Categorical(action_probs[0])
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        
        # Store communication for next step
        self.last_communication = communication_message[0].detach()
        self.communication_history.append(self.last_communication.clone())
        
        return action.item(), communication_message[0], log_prob, action_probs[0]
    
    def _aggregate_messages(self, messages: List[torch.Tensor]) -> torch.Tensor:
        """Aggregate received communication messages."""
        if not messages:
            return torch.zeros(self.communication_dim).to(self.device)
        
        # Simple averaging aggregation
        aggregated = torch.stack(messages).mean(dim=0)
        return aggregated
    
    def store_transition(self, observation: Dict, action: int, reward: float, 
                        message: torch.Tensor, log_prob: torch.Tensor, value: float):
        """Store transition data for episode."""
        self.episode_data['observations'].append(observation)
        self.episode_data['actions'].append(action)
        self.episode_data['rewards'].append(reward)
        self.episode_data['messages'].append(message)
        self.episode_data['log_probs'].append(log_prob)
        self.episode_data['values'].append(value)
    
    def update(self, critic_values: List[float], baseline_values: List[float]) -> Dict[str, float]:
        """
        Update actor using COMA with communication.
        
        Args:
            critic_values: Q-values from centralized critic
            baseline_values: Baseline values for counterfactual advantage
            
        Returns:
            Dictionary containing loss metrics
        """
        if len(self.episode_data['rewards']) == 0:
            return {'actor_loss': 0.0}
        
        # Compute TD(λ) targets
        td_targets = self._compute_td_lambda_targets(critic_values)
        
        # Compute advantages
        advantages = []
        for i in range(len(td_targets)):
            advantage = td_targets[i] - baseline_values[i]
            advantages.append(advantage)
        
        # Convert to tensors
        log_probs = torch.stack(self.episode_data['log_probs'])
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        
        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Actor loss (policy gradient with communication regularization)
        actor_loss = -(log_probs * advantages).mean()
        
        # Add communication regularization
        comm_reg = 0.0
        for message in self.episode_data['messages']:
            comm_reg += torch.norm(message, p=2)
        comm_reg = comm_reg / len(self.episode_data['messages'])
        
        total_loss = actor_loss + 0.01 * comm_reg
        
        # Update actor
        self.actor_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()
        
        # Clear episode data
        self.clear_episode_data()
        
        return {
            'actor_loss': actor_loss.item(),
            'comm_regularization': comm_reg.item(),
            'advantage_mean': advantages.mean().item(),
            'advantage_std': advantages.std().item()
        }
    
    def _compute_td_lambda_targets(self, critic_values: List[float]) -> List[float]:
        """Compute TD(λ) targets."""
        rewards = self.episode_data['rewards']
        values = critic_values
        
        td_targets = []
        
        for t in range(len(rewards)):
            td_target = 0.0
            discount = 1.0
            
            for k in range(t, len(rewards)):
                if k == len(rewards) - 1:
                    # Terminal step
                    td_error = rewards[k] - values[k]
                else:
                    td_error = rewards[k] + self.gamma * values[k + 1] - values[k]
                
                td_target += discount * td_error
                discount *= self.gamma * self.td_lambda
                
                if k < len(rewards) - 1:
                    td_target += discount * values[k + 1]
                    break
            
            td_targets.append(values[t] + td_target)
        
        return td_targets
    
    def clear_episode_data(self):
        """Clear episode data."""
        self.episode_data = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'messages': [],
            'log_probs': [],
            'values': []
        }
    
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
    
    def get_communication_message(self) -> torch.Tensor:
        """Get the last communication message."""
        return self.last_communication
    
    def save_model(self, path: str):
        """Save the agent's model."""
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'last_communication': self.last_communication,
            'communication_history': list(self.communication_history),
            'agent_id': self.agent_id,
            'config': self.config
        }
        torch.save(checkpoint, f"{path}_comacomm_agent_{self.agent_id}.pth")
        print(f"Saved COMA-Comm Agent {self.agent_id} model to {path}_comacomm_agent_{self.agent_id}.pth")
    
    def load_model(self, path: str):
        """Load the agent's model."""
        checkpoint = torch.load(f"{path}_comacomm_agent_{self.agent_id}.pth", map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.last_communication = checkpoint['last_communication']
        self.communication_history = deque(checkpoint['communication_history'], 
                                         maxlen=self.config.get('comm_history_length', 10))
        
        print(f"Loaded COMA-Comm Agent {self.agent_id} model from {path}_comacomm_agent_{self.agent_id}.pth")


class COMACommActor(nn.Module):
    """Actor network with communication generation for COMA-Comm."""
    
    def __init__(self, obs_space: Dict, action_space: int, communication_dim: int, config: Dict):
        super().__init__()
        
        self.obs_dim = self._estimate_obs_dim(obs_space)
        self.action_space = action_space
        self.communication_dim = communication_dim
        
        hidden_dim = config.get('hidden_dim', 128)
        
        # Observation encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(self.obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Communication encoder
        self.comm_encoder = nn.Sequential(
            nn.Linear(communication_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Combined feature processor
        self.feature_processor = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Action head
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, action_space),
            nn.Softmax(dim=-1)
        )
        
        # Communication head
        self.communication_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, communication_dim),
            nn.Tanh()  # Normalize communication messages
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
    
    def forward(self, obs: torch.Tensor, received_message: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through actor network.
        
        Args:
            obs: Observation tensor
            received_message: Received communication message
            
        Returns:
            Tuple of (action_probabilities, communication_message)
        """
        # Encode observation
        obs_features = self.obs_encoder(obs)
        
        # Encode received communication
        comm_features = self.comm_encoder(received_message)
        
        # Combine features
        combined_features = torch.cat([obs_features, comm_features], dim=-1)
        processed_features = self.feature_processor(combined_features)
        
        # Generate action probabilities
        action_probs = self.action_head(processed_features)
        
        # Generate communication message
        communication_message = self.communication_head(processed_features)
        
        return action_probs, communication_message


class COMACommCritic(nn.Module):
    """Centralized critic network with communication awareness for COMA-Comm."""
    
    def __init__(self, obs_space: Dict, action_space: int, communication_dim: int, config: Dict):
        super().__init__()
        
        self.obs_dim = self._estimate_obs_dim(obs_space)
        self.action_space = action_space
        self.communication_dim = communication_dim
        
        hidden_dim = config.get('hidden_dim', 256)
        
        # This will be set by the algorithm based on number of agents
        self.n_agents = config.get('n_agents', 2)
        
        # Global state encoder (all agents' observations)
        total_obs_dim = self.obs_dim * self.n_agents
        total_comm_dim = communication_dim * self.n_agents
        
        self.state_encoder = nn.Sequential(
            nn.Linear(total_obs_dim + total_comm_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Action encoder for joint actions
        self.action_encoder = nn.Sequential(
            nn.Linear(self.n_agents, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Q-value head
        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
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
    
    def forward(self, all_obs: torch.Tensor, all_actions: torch.Tensor, 
               all_messages: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through centralized critic.
        
        Args:
            all_obs: All agents' observations [batch_size, n_agents, obs_dim]
            all_actions: All agents' actions [batch_size, n_agents]
            all_messages: All agents' communication messages [batch_size, n_agents, comm_dim]
            
        Returns:
            Q-value for the joint action
        """
        batch_size = all_obs.shape[0]
        
        # Flatten observations and messages
        flat_obs = all_obs.view(batch_size, -1)
        flat_messages = all_messages.view(batch_size, -1)
        
        # Encode global state
        global_state = torch.cat([flat_obs, flat_messages], dim=-1)
        state_features = self.state_encoder(global_state)
        
        # Encode joint actions
        action_features = self.action_encoder(all_actions.float())
        
        # Combine and get Q-value
        combined_features = torch.cat([state_features, action_features], dim=-1)
        q_value = self.q_head(combined_features)
        
        return q_value
    
    def get_counterfactual_values(self, all_obs: torch.Tensor, all_actions: torch.Tensor,
                                 all_messages: torch.Tensor, agent_id: int) -> torch.Tensor:
        """
        Get counterfactual Q-values for credit assignment.
        
        Args:
            all_obs: All agents' observations
            all_actions: All agents' actions
            all_messages: All agents' communication messages
            agent_id: ID of the agent to compute counterfactuals for
            
        Returns:
            Q-values for all possible actions of the specified agent
        """
        batch_size = all_obs.shape[0]
        counterfactual_values = []
        
        for action in range(self.action_space):
            # Create counterfactual joint action
            counterfactual_actions = all_actions.clone()
            counterfactual_actions[:, agent_id] = action
            
            # Get Q-value for this counterfactual
            q_value = self.forward(all_obs, counterfactual_actions, all_messages)
            counterfactual_values.append(q_value)
        
        return torch.cat(counterfactual_values, dim=-1)


class COMAComm(MARLAlgorithm):
    """
    Counterfactual Multi-Agent Policy Gradients with Communication.
    
    Extends COMA with explicit communication channels for improved
    credit assignment and coordination.
    """
    
    def __init__(self, env, config: Dict, device: torch.device):
        """
        Initialize COMA-Comm algorithm.
        
        Args:
            env: Multi-agent environment
            config: Configuration dictionary
            device: PyTorch device
        """
        super().__init__(env, config, device)
        
        # COMA-Comm specific parameters
        self.train_start = config.get('train_start', 1000)
        self.critic_update_freq = config.get('critic_update_freq', 1)
        
        # Set number of agents in config for critic
        self.config['n_agents'] = self.n_agents
        
        # Centralized critic
        obs_space = self._get_obs_space()
        action_space = len(self.env.actions)
        communication_dim = config.get('communication_dim', 32)
        
        self.critic = COMACommCritic(obs_space, action_space, communication_dim, self.config).to(self.device)
        self.target_critic = COMACommCritic(obs_space, action_space, communication_dim, self.config).to(self.device)
        
        # Copy weights to target critic
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # Critic optimizer
        self.critic_optimizer = Adam(self.critic.parameters(), lr=config.get('critic_lr', 1e-3))
        
        # Episode buffer
        self.episode_buffer = []
        
        print(f"Initialized COMA-Comm")
        print(f"Number of agents: {self.n_agents}")
        print(f"Critic parameters: {sum(p.numel() for p in self.critic.parameters())}")
    
    def _create_agents(self):
        """Create COMA-Comm agents."""
        obs_space = self._get_obs_space()
        action_space = len(self.env.actions)
        
        for i in range(self.n_agents):
            agent = COMACommAgent(i, obs_space, action_space, self.config)
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
        """Collect rollout with communication between agents."""
        obs = env.reset()
        done = False
        step_count = 0
        total_reward = [0.0] * self.n_agents
        
        # Initialize communication messages
        agent_messages = [torch.zeros(self.config.get('communication_dim', 32)).to(self.device) 
                         for _ in range(self.n_agents)]
        
        episode_transitions = []
        
        while not done:
            # Get all observations
            all_observations = []
            for i in range(self.n_agents):
                agent_obs = self._extract_agent_obs(obs, i)
                all_observations.append(agent_obs)
            
            # Get communication messages from previous step
            prev_messages = [agent.get_communication_message() for agent in self.agents]
            
            # Get actions and new communication messages
            actions = []
            new_messages = []
            log_probs = []
            
            for i, agent in enumerate(self.agents):
                # Get messages from other agents (excluding self)
                received_messages = [prev_messages[j] for j in range(self.n_agents) if j != i]
                
                action, comm_msg, log_prob, action_probs = agent.get_action(
                    all_observations[i], received_messages, training=True
                )
                actions.append(action)
                new_messages.append(comm_msg)
                log_probs.append(log_prob)
            
            # Take environment step
            next_obs, rewards, done, info = env.step(actions)
            
            # Store transition
            reward = rewards[0] if isinstance(rewards, list) else rewards  # Shared reward
            for i in range(self.n_agents):
                total_reward[i] += reward
            
            transition = {
                'all_observations': all_observations,
                'all_actions': actions,
                'reward': reward,
                'all_messages': prev_messages,
                'log_probs': log_probs,
                'done': done
            }
            
            episode_transitions.append(transition)
            
            # Store individual transitions in agents
            for i, agent in enumerate(self.agents):
                agent.store_transition(
                    all_observations[i], actions[i], reward, 
                    prev_messages[i], log_probs[i], 0.0  # Value will be computed later
                )
            
            # Update communication messages
            agent_messages = new_messages
            obs = next_obs
            step_count += 1
            self.total_steps += 1
        
        # Store episode
        self.episode_buffer.append(episode_transitions)
        
        return {
            'episode_length': step_count,
            'total_reward': total_reward
        }
    
    def train_step(self, rollout_data: Dict) -> Dict[str, float]:
        """Perform COMA-Comm training step."""
        if self.total_steps < self.train_start or len(self.episode_buffer) == 0:
            return {
                'episode_length': rollout_data['episode_length'],
                'total_reward': sum(rollout_data['total_reward']) / self.n_agents
            }
        
        # Train on the latest episode
        episode = self.episode_buffer[-1]
        
        # Update critic
        critic_loss = self._update_critic(episode)
        
        # Get critic values and baselines for the episode
        critic_values, baseline_values = self._get_critic_values(episode)
        
        # Update actors
        actor_losses = []
        for i, agent in enumerate(self.agents):
            agent_critic_values = [v[i] for v in critic_values]
            agent_baseline_values = [v[i] for v in baseline_values]
            
            loss_info = agent.update(agent_critic_values, agent_baseline_values)
            actor_losses.append(loss_info)
        
        # Clear episode buffer periodically
        if len(self.episode_buffer) > 10:
            self.episode_buffer = self.episode_buffer[-5:]
        
        if actor_losses:
            # Average metrics across agents
            avg_metrics = {}
            for key in actor_losses[0].keys():
                avg_metrics[f'avg_{key}'] = np.mean([loss[key] for loss in actor_losses])
            
            avg_metrics.update({
                'critic_loss': critic_loss,
                'episode_length': rollout_data['episode_length'],
                'total_reward': sum(rollout_data['total_reward']) / self.n_agents,
                'total_steps': self.total_steps
            })
            
            return avg_metrics
        
        return {
            'critic_loss': critic_loss,
            'episode_length': rollout_data['episode_length'],
            'total_reward': sum(rollout_data['total_reward']) / self.n_agents,
            'total_steps': self.total_steps
        }
    
    def _update_critic(self, episode: List[Dict]) -> float:
        """Update centralized critic."""
        total_loss = 0.0
        
        for t, transition in enumerate(episode):
            # Prepare inputs
            all_obs = self._prepare_observations(transition['all_observations'])
            all_actions = torch.tensor(transition['all_actions'], dtype=torch.long).unsqueeze(0).to(self.device)
            all_messages = torch.stack(transition['all_messages']).unsqueeze(0).to(self.device)
            
            # Current Q-value
            current_q = self.critic(all_obs, all_actions, all_messages)
            
            # Target Q-value
            with torch.no_grad():
                if t < len(episode) - 1:
                    next_transition = episode[t + 1]
                    next_all_obs = self._prepare_observations(next_transition['all_observations'])
                    next_all_actions = torch.tensor(next_transition['all_actions'], dtype=torch.long).unsqueeze(0).to(self.device)
                    next_all_messages = torch.stack(next_transition['all_messages']).unsqueeze(0).to(self.device)
                    
                    next_q = self.target_critic(next_all_obs, next_all_actions, next_all_messages)
                    target = transition['reward'] + self.config.get('gamma', 0.99) * next_q
                else:
                    target = torch.tensor([[transition['reward']]], dtype=torch.float32).to(self.device)
            
            # Critic loss
            loss = F.mse_loss(current_q, target)
            total_loss += loss.item()
            
            # Update critic
            self.critic_optimizer.zero_grad()
            loss.backward()
            self.critic_optimizer.step()
        
        # Update target critic
        if self.total_steps % 100 == 0:
            self.target_critic.load_state_dict(self.critic.state_dict())
        
        return total_loss / len(episode)
    
    def _get_critic_values(self, episode: List[Dict]) -> Tuple[List[List[float]], List[List[float]]]:
        """Get critic values and baseline values for the episode."""
        critic_values = []
        baseline_values = []
        
        with torch.no_grad():
            for transition in episode:
                all_obs = self._prepare_observations(transition['all_observations'])
                all_actions = torch.tensor(transition['all_actions'], dtype=torch.long).unsqueeze(0).to(self.device)
                all_messages = torch.stack(transition['all_messages']).unsqueeze(0).to(self.device)
                
                # Get joint Q-value
                joint_q = self.critic(all_obs, all_actions, all_messages).item()
                
                # Get counterfactual baselines for each agent
                agent_baselines = []
                agent_values = []
                
                for i in range(self.n_agents):
                    # Get counterfactual Q-values
                    counterfactual_qs = self.critic.get_counterfactual_values(
                        all_obs, all_actions, all_messages, i
                    )[0]  # Remove batch dimension
                    
                    # Baseline is the value when taking the current action
                    baseline = counterfactual_qs[transition['all_actions'][i]].item()
                    agent_baselines.append(baseline)
                    agent_values.append(joint_q)  # Use joint Q-value
                
                critic_values.append(agent_values)
                baseline_values.append(agent_baselines)
        
        return critic_values, baseline_values
    
    def _prepare_observations(self, observations: List[Dict]) -> torch.Tensor:
        """Prepare observations tensor."""
        obs_tensors = []
        for obs in observations:
            if isinstance(obs, dict):
                obs_list = []
                for key, value in obs.items():
                    if isinstance(value, np.ndarray):
                        obs_list.append(torch.tensor(value, dtype=torch.float32).flatten())
                    else:
                        obs_list.append(torch.tensor([value], dtype=torch.float32))
                obs_tensor = torch.cat(obs_list)
            else:
                obs_tensor = torch.tensor(obs, dtype=torch.float32)
            
            obs_tensors.append(obs_tensor)
        
        return torch.stack(obs_tensors).unsqueeze(0).to(self.device)
