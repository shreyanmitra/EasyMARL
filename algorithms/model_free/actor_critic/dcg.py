"""
Deep Coordination Graphs (DCG) algorithm.

DCG factorizes the joint Q-function over a coordination graph structure,
enabling scalable coordination with pairwise interactions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from typing import Dict, List, Tuple, Any, Set
from collections import deque
import random
import networkx as nx

from .base import MARLAgent, MARLAlgorithm


class DCGAgent(MARLAgent):
    """
    DCG agent with coordination graph structure.
    """
    
    def __init__(self, agent_id: int, obs_space: Dict, action_space: int, config: Dict):
        """
        Initialize the DCG agent.
        
        Args:
            agent_id: Unique identifier for this agent
            obs_space: Individual observation space
            action_space: Number of available actions
            config: Configuration dictionary
        """
        super().__init__(agent_id, obs_space, action_space, config)
        
        # DCG hyperparameters
        self.gamma = config.get('gamma', 0.99)
        self.epsilon_start = config.get('epsilon_start', 1.0)
        self.epsilon_end = config.get('epsilon_end', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.epsilon = self.epsilon_start
        
        # Coordination graph structure
        self.neighbors = set()  # Will be set by the main algorithm
        self.coordination_edges = {}  # Edge relationships
        
        # Individual Q-network
        self.individual_q_network = DCGIndividualQNetwork(obs_space, action_space, config).to(self.device)
        self.target_individual_q_network = DCGIndividualQNetwork(obs_space, action_space, config).to(self.device)
        
        # Pairwise Q-networks for coordination with neighbors
        self.pairwise_q_networks = nn.ModuleDict()
        self.target_pairwise_q_networks = nn.ModuleDict()
        
        # Copy weights to target networks
        self.target_individual_q_network.load_state_dict(self.individual_q_network.state_dict())
        
        # Optimizer
        self.optimizer = Adam(
            list(self.individual_q_network.parameters()) + 
            list(self.pairwise_q_networks.parameters()), 
            lr=config.get('learning_rate', 1e-3)
        )
        
        print(f"Initialized DCG Agent {agent_id}")
        print(f"Individual Q-network parameters: {sum(p.numel() for p in self.individual_q_network.parameters())}")
    
    def add_neighbor(self, neighbor_id: int, obs_space: Dict, action_space: int, config: Dict):
        """Add a neighbor and create pairwise Q-network."""
        self.neighbors.add(neighbor_id)
        
        # Create pairwise Q-network for this edge
        edge_key = f"{min(self.agent_id, neighbor_id)}_{max(self.agent_id, neighbor_id)}"
        
        if edge_key not in self.pairwise_q_networks:
            pairwise_net = DCGPairwiseQNetwork(obs_space, action_space, config).to(self.device)
            target_pairwise_net = DCGPairwiseQNetwork(obs_space, action_space, config).to(self.device)
            target_pairwise_net.load_state_dict(pairwise_net.state_dict())
            
            self.pairwise_q_networks[edge_key] = pairwise_net
            self.target_pairwise_q_networks[edge_key] = target_pairwise_net
            
            print(f"Agent {self.agent_id}: Added pairwise network for edge {edge_key}")
    
    def get_action(self, observation: Dict, neighbor_observations: Dict = None, training: bool = True) -> Tuple[int, float]:
        """
        Select an action using coordination with neighbors.
        
        Args:
            observation: Individual agent observation
            neighbor_observations: Observations of neighboring agents
            training: Whether in training mode
            
        Returns:
            Tuple of (action, q_value)
        """
        with torch.no_grad():
            if training and random.random() < self.epsilon:
                action = random.randint(0, self.action_space - 1)
                q_value = 0.0
            else:
                # Compute coordinated action using max-plus algorithm
                action, q_value = self._coordinate_action(observation, neighbor_observations)
            
            return action, q_value
    
    def _coordinate_action(self, observation: Dict, neighbor_observations: Dict = None) -> Tuple[int, float]:
        """Coordinate action selection using max-plus algorithm."""
        obs_tensor = self._process_observation(observation)
        
        # Get individual Q-values
        individual_q = self.individual_q_network(obs_tensor)[0]  # [action_space]
        
        # Initialize messages for max-plus
        total_utilities = individual_q.clone()
        
        # Add pairwise coordination utilities
        if neighbor_observations:
            for neighbor_id in self.neighbors:
                if neighbor_id in neighbor_observations:
                    neighbor_obs = neighbor_observations[neighbor_id]
                    neighbor_obs_tensor = self._process_observation(neighbor_obs)
                    
                    # Get edge key
                    edge_key = f"{min(self.agent_id, neighbor_id)}_{max(self.agent_id, neighbor_id)}"
                    
                    if edge_key in self.pairwise_q_networks:
                        # Get pairwise Q-values
                        pairwise_q = self.pairwise_q_networks[edge_key](obs_tensor, neighbor_obs_tensor)
                        
                        # Add coordination utility (simplified max-plus)
                        if self.agent_id < neighbor_id:
                            # This agent is first in the edge
                            coord_utility = torch.max(pairwise_q, dim=1)[0]  # Max over neighbor actions
                        else:
                            # This agent is second in the edge
                            coord_utility = torch.max(pairwise_q, dim=0)[0]  # Max over neighbor actions
                        
                        total_utilities += coord_utility
        
        # Select best action
        best_action = torch.argmax(total_utilities).item()
        best_value = total_utilities[best_action].item()
        
        return best_action, best_value
    
    def get_individual_q_values(self, observation: Dict) -> torch.Tensor:
        """Get individual Q-values."""
        obs_tensor = self._process_observation(observation)
        return self.individual_q_network(obs_tensor)
    
    def get_pairwise_q_values(self, observation: Dict, neighbor_observation: Dict, neighbor_id: int) -> torch.Tensor:
        """Get pairwise Q-values with a specific neighbor."""
        edge_key = f"{min(self.agent_id, neighbor_id)}_{max(self.agent_id, neighbor_id)}"
        
        if edge_key in self.pairwise_q_networks:
            obs_tensor = self._process_observation(observation)
            neighbor_obs_tensor = self._process_observation(neighbor_observation)
            return self.pairwise_q_networks[edge_key](obs_tensor, neighbor_obs_tensor)
        else:
            return torch.zeros(self.action_space, self.action_space).to(self.device)
    
    def update(self, batch_data: Dict) -> Dict[str, float]:
        """
        Update individual and pairwise Q-networks.
        
        Args:
            batch_data: Dictionary containing transitions
            
        Returns:
            Dictionary containing loss metrics
        """
        observations = batch_data['observations']
        actions = batch_data['actions']
        rewards = batch_data['rewards']
        next_observations = batch_data['next_observations']
        dones = batch_data['dones']
        neighbor_data = batch_data.get('neighbor_data', {})
        
        total_individual_loss = 0.0
        total_pairwise_loss = 0.0
        
        for i in range(len(observations)):
            # Update individual Q-network
            individual_loss = self._update_individual_q(
                observations[i], actions[i], rewards[i], next_observations[i], dones[i]
            )
            total_individual_loss += individual_loss
            
            # Update pairwise Q-networks
            if neighbor_data:
                pairwise_loss = self._update_pairwise_q(
                    observations[i], actions[i], rewards[i], 
                    next_observations[i], dones[i], neighbor_data
                )
                total_pairwise_loss += pairwise_loss
        
        return {
            'individual_q_loss': total_individual_loss / len(observations),
            'pairwise_q_loss': total_pairwise_loss / len(observations),
            'epsilon': self.epsilon
        }
    
    def _update_individual_q(self, observation: Dict, action: int, reward: float,
                           next_observation: Dict, done: bool) -> float:
        """Update individual Q-network."""
        obs_tensor = self._process_observation(observation)
        next_obs_tensor = self._process_observation(next_observation)
        
        # Current Q-value
        current_q = self.individual_q_network(obs_tensor)[0, action]
        
        # Target Q-value
        with torch.no_grad():
            if not done:
                next_q = self.target_individual_q_network(next_obs_tensor)
                max_next_q = torch.max(next_q)
                target = reward + self.gamma * max_next_q
            else:
                target = torch.tensor(reward).to(self.device)
        
        # Individual Q-learning loss
        loss = F.mse_loss(current_q, target)
        
        return loss.item()
    
    def _update_pairwise_q(self, observation: Dict, action: int, reward: float,
                          next_observation: Dict, done: bool, neighbor_data: Dict) -> float:
        """Update pairwise Q-networks."""
        total_loss = 0.0
        update_count = 0
        
        for neighbor_id in self.neighbors:
            if neighbor_id in neighbor_data:
                neighbor_obs = neighbor_data[neighbor_id]['observation']
                neighbor_action = neighbor_data[neighbor_id]['action']
                neighbor_next_obs = neighbor_data[neighbor_id]['next_observation']
                
                edge_key = f"{min(self.agent_id, neighbor_id)}_{max(self.agent_id, neighbor_id)}"
                
                if edge_key in self.pairwise_q_networks:
                    obs_tensor = self._process_observation(observation)
                    neighbor_obs_tensor = self._process_observation(neighbor_obs)
                    next_obs_tensor = self._process_observation(next_observation)
                    next_neighbor_obs_tensor = self._process_observation(neighbor_next_obs)
                    
                    # Current pairwise Q-value
                    current_pairwise_q = self.pairwise_q_networks[edge_key](obs_tensor, neighbor_obs_tensor)
                    
                    if self.agent_id < neighbor_id:
                        current_q_value = current_pairwise_q[action, neighbor_action]
                    else:
                        current_q_value = current_pairwise_q[neighbor_action, action]
                    
                    # Target pairwise Q-value
                    with torch.no_grad():
                        if not done:
                            next_pairwise_q = self.target_pairwise_q_networks[edge_key](
                                next_obs_tensor, next_neighbor_obs_tensor
                            )
                            max_next_q = torch.max(next_pairwise_q)
                            target = reward + self.gamma * max_next_q
                        else:
                            target = torch.tensor(reward).to(self.device)
                    
                    # Pairwise Q-learning loss
                    loss = F.mse_loss(current_q_value, target)
                    total_loss += loss.item()
                    update_count += 1
        
        return total_loss / max(1, update_count)
    
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
    
    def update_target_networks(self):
        """Update target networks."""
        self.target_individual_q_network.load_state_dict(self.individual_q_network.state_dict())
        
        for edge_key in self.pairwise_q_networks:
            self.target_pairwise_q_networks[edge_key].load_state_dict(
                self.pairwise_q_networks[edge_key].state_dict()
            )
    
    def decay_epsilon(self):
        """Decay epsilon for exploration."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def reset_memory(self):
        """Reset agent memory."""
        self.memory = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'next_observations': [],
            'dones': [],
            'neighbor_data': []
        }
    
    def save_model(self, path: str):
        """Save the agent's model."""
        checkpoint = {
            'individual_q_network_state_dict': self.individual_q_network.state_dict(),
            'target_individual_q_network_state_dict': self.target_individual_q_network.state_dict(),
            'pairwise_q_networks_state_dict': {k: v.state_dict() for k, v in self.pairwise_q_networks.items()},
            'target_pairwise_q_networks_state_dict': {k: v.state_dict() for k, v in self.target_pairwise_q_networks.items()},
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'neighbors': list(self.neighbors),
            'agent_id': self.agent_id,
            'config': self.config
        }
        torch.save(checkpoint, f"{path}_dcg_agent_{self.agent_id}.pth")
        print(f"Saved DCG Agent {self.agent_id} model to {path}_dcg_agent_{self.agent_id}.pth")
    
    def load_model(self, path: str):
        """Load the agent's model."""
        checkpoint = torch.load(f"{path}_dcg_agent_{self.agent_id}.pth", map_location=self.device)
        
        self.individual_q_network.load_state_dict(checkpoint['individual_q_network_state_dict'])
        self.target_individual_q_network.load_state_dict(checkpoint['target_individual_q_network_state_dict'])
        
        for k, v in checkpoint['pairwise_q_networks_state_dict'].items():
            if k in self.pairwise_q_networks:
                self.pairwise_q_networks[k].load_state_dict(v)
        
        for k, v in checkpoint['target_pairwise_q_networks_state_dict'].items():
            if k in self.target_pairwise_q_networks:
                self.target_pairwise_q_networks[k].load_state_dict(v)
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.neighbors = set(checkpoint['neighbors'])
        
        print(f"Loaded DCG Agent {self.agent_id} model from {path}_dcg_agent_{self.agent_id}.pth")


class DCGIndividualQNetwork(nn.Module):
    """Individual Q-network for DCG agent."""
    
    def __init__(self, obs_space: Dict, action_space: int, config: Dict):
        super().__init__()
        
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
        """Forward pass through individual Q-network."""
        return self.network(obs)


class DCGPairwiseQNetwork(nn.Module):
    """Pairwise Q-network for coordination between two agents."""
    
    def __init__(self, obs_space: Dict, action_space: int, config: Dict):
        super().__init__()
        
        self.obs_dim = self._estimate_obs_dim(obs_space)
        self.action_space = action_space
        hidden_dim = config.get('hidden_dim', 128)
        
        # Observation encoders
        self.obs_encoder1 = nn.Sequential(
            nn.Linear(self.obs_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.obs_encoder2 = nn.Sequential(
            nn.Linear(self.obs_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Pairwise Q-value network
        self.pairwise_network = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_space * action_space)
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
    
    def forward(self, obs1: torch.Tensor, obs2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through pairwise Q-network.
        
        Args:
            obs1: Observation of first agent
            obs2: Observation of second agent
            
        Returns:
            Pairwise Q-values [action_space, action_space]
        """
        # Encode observations
        encoded_obs1 = self.obs_encoder1(obs1)
        encoded_obs2 = self.obs_encoder2(obs2)
        
        # Combine observations
        combined = torch.cat([encoded_obs1, encoded_obs2], dim=-1)
        
        # Get pairwise Q-values
        pairwise_q = self.pairwise_network(combined)
        
        # Reshape to [batch_size, action_space, action_space]
        batch_size = obs1.shape[0]
        pairwise_q = pairwise_q.view(batch_size, self.action_space, self.action_space)
        
        return pairwise_q


class DCGReplayBuffer:
    """Replay buffer for DCG with neighbor information."""
    
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
            'actions': [t['action'] for t in batch],
            'rewards': [t['reward'] for t in batch],
            'next_observations': [t['next_observation'] for t in batch],
            'dones': [t['done'] for t in batch],
            'neighbor_data': [t.get('neighbor_data', {}) for t in batch]
        }
        
        return batch_data
    
    def __len__(self):
        return len(self.buffer)


class DCG(MARLAlgorithm):
    """
    Deep Coordination Graphs (DCG) algorithm.
    
    Factorizes joint Q-function over coordination graph structure
    with pairwise interactions between neighboring agents.
    """
    
    def __init__(self, env, config: Dict, device: torch.device):
        """
        Initialize DCG algorithm.
        
        Args:
            env: Multi-agent environment
            config: Configuration dictionary
            device: PyTorch device
        """
        super().__init__(env, config, device)
        
        # DCG specific parameters
        self.batch_size = config.get('batch_size', 32)
        self.memory_size = config.get('memory_size', 10000)
        self.train_start = config.get('train_start', 1000)
        self.update_interval = config.get('update_interval', 1)
        self.target_update_freq = config.get('target_update_freq', 100)
        
        # Coordination graph structure
        self.coordination_graph = self._create_coordination_graph(config)
        
        # Create replay buffers for each agent
        self.replay_buffers = [
            DCGReplayBuffer(self.memory_size) for _ in range(self.n_agents)
        ]
        
        print(f"Initialized DCG")
        print(f"Coordination graph edges: {list(self.coordination_graph.edges())}")
        print(f"Batch size: {self.batch_size}")
    
    def _create_coordination_graph(self, config: Dict) -> nx.Graph:
        """Create coordination graph structure."""
        graph_type = config.get('graph_type', 'complete')
        
        G = nx.Graph()
        G.add_nodes_from(range(self.n_agents))
        
        if graph_type == 'complete':
            # Fully connected graph
            for i in range(self.n_agents):
                for j in range(i + 1, self.n_agents):
                    G.add_edge(i, j)
        elif graph_type == 'star':
            # Star graph with agent 0 at center
            for i in range(1, self.n_agents):
                G.add_edge(0, i)
        elif graph_type == 'line':
            # Line graph
            for i in range(self.n_agents - 1):
                G.add_edge(i, i + 1)
        elif graph_type == 'cycle':
            # Cycle graph
            for i in range(self.n_agents - 1):
                G.add_edge(i, i + 1)
            if self.n_agents > 2:
                G.add_edge(self.n_agents - 1, 0)
        else:
            # Default to complete graph
            for i in range(self.n_agents):
                for j in range(i + 1, self.n_agents):
                    G.add_edge(i, j)
        
        return G
    
    def _create_agents(self):
        """Create DCG agents with coordination graph structure."""
        obs_space = self._get_obs_space()
        action_space = len(self.env.actions)
        
        # Create agents
        for i in range(self.n_agents):
            agent = DCGAgent(i, obs_space, action_space, self.config)
            self.agents.append(agent)
        
        # Set up coordination graph edges
        for i, j in self.coordination_graph.edges():
            self.agents[i].add_neighbor(j, obs_space, action_space, self.config)
            self.agents[j].add_neighbor(i, obs_space, action_space, self.config)
    
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
        """Collect transitions with neighbor information."""
        obs = env.reset()
        done = False
        step_count = 0
        total_reward = [0.0] * self.n_agents
        
        while not done:
            # Get observations for all agents
            agent_observations = {}
            for i in range(self.n_agents):
                agent_observations[i] = self._extract_agent_obs(obs, i)
            
            # Get actions from all agents with coordination
            actions = []
            for i, agent in enumerate(self.agents):
                agent_obs = agent_observations[i]
                
                # Get neighbor observations
                neighbor_obs = {}
                for neighbor_id in agent.neighbors:
                    neighbor_obs[neighbor_id] = agent_observations[neighbor_id]
                
                action, _ = agent.get_action(agent_obs, neighbor_obs, training=True)
                actions.append(action)
            
            # Take environment step
            next_obs, rewards, done, info = env.step(actions)
            
            # Store transitions with neighbor data
            for i in range(self.n_agents):
                agent_obs = agent_observations[i]
                next_agent_obs = self._extract_agent_obs(next_obs, i)
                
                reward = rewards[i] if isinstance(rewards, list) else rewards
                total_reward[i] += reward
                
                # Collect neighbor data
                neighbor_data = {}
                for neighbor_id in self.agents[i].neighbors:
                    neighbor_data[neighbor_id] = {
                        'observation': agent_observations[neighbor_id],
                        'action': actions[neighbor_id],
                        'next_observation': self._extract_agent_obs(next_obs, neighbor_id)
                    }
                
                transition = {
                    'observation': agent_obs,
                    'action': actions[i],
                    'reward': reward,
                    'next_observation': next_agent_obs,
                    'done': done,
                    'neighbor_data': neighbor_data
                }
                
                self.replay_buffers[i].add(transition)
            
            obs = next_obs
            step_count += 1
            self.total_steps += 1
        
        # Decay epsilon for all agents
        for agent in self.agents:
            agent.decay_epsilon()
        
        return {
            'episode_length': step_count,
            'total_reward': total_reward
        }
    
    def train_step(self, rollout_data: Dict) -> Dict[str, float]:
        """Perform DCG training step."""
        if self.total_steps < self.train_start:
            return {
                'episode_length': rollout_data['episode_length'],
                'total_reward': sum(rollout_data['total_reward']) / self.n_agents
            }
        
        # Train each agent if we have enough experience
        if self.total_steps % self.update_interval == 0:
            agent_losses = []
            
            for i, agent in enumerate(self.agents):
                if len(self.replay_buffers[i]) >= self.batch_size:
                    # Sample batch and update agent
                    batch_data = self.replay_buffers[i].sample(self.batch_size)
                    loss_info = agent.update(batch_data)
                    agent_losses.append(loss_info)
            
            # Update target networks
            if self.total_steps % self.target_update_freq == 0:
                for agent in self.agents:
                    agent.update_target_networks()
                print(f"Updated target networks at step {self.total_steps}")
            
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
