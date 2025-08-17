"""
(C) Shreyan Mitra, based on starter code by Natasha Jaques

Neural Fictitious Self-Play (NFSP) Algorithm for Multi-Agent Reinforcement Learning

NFSP is an advanced algorithm that combines:
1. Deep Q-Network (DQN) for learning best responses to opponent strategies
2. Supervised learning to learn average behavior policies
3. Self-play mechanism to approximate Nash equilibria in games

This implementation is designed to be educational for MARL beginners while
maintaining research-level functionality.

Key Concepts for Beginners:
- Fictitious Play: Classical game theory concept where agents learn by assuming
  opponents play according to historical average strategies
- Neural Networks: Used to approximate both Q-functions and average policies
- Self-Play: Agents train against copies of themselves to discover equilibria
- Nash Equilibrium: Solution concept where no agent can improve by changing strategy

Paper: "Neural Fictitious Self-Play in Imperfect Information Games" (Heinrich & Silver, 2016)
Use Cases: Two-player games, poker, competitive multi-agent scenarios
"""

# Import necessary libraries for deep learning and multi-agent systems
import torch                    # PyTorch for neural networks and automatic differentiation
import torch.nn as nn           # Neural network modules and loss functions
import torch.nn.functional as F # Activation functions and other neural network utilities
from torch.optim import Adam    # Adam optimizer for gradient-based learning
import numpy as np              # Numerical computations and array operations
from typing import Dict, List, Tuple, Any  # Type hints for better code clarity
from collections import deque   # Efficient queue data structure for experience replay
import random                   # Random number generation for exploration

# Import base classes from our MARL framework
from .base import MARLAgent, MARLAlgorithm


class NFSPAgent(MARLAgent):
    """
    Neural Fictitious Self-Play Agent
    
    This agent implements the NFSP algorithm, which learns to play games by:
    1. Learning best responses to opponents using Deep Q-Learning
    2. Learning average behavior policies using supervised learning
    3. Mixing between these two policies during play
    
    The agent maintains two neural networks:
    - Q-network: Learns optimal actions against current opponent strategy
    - Policy network: Learns average behavior over all past strategies
    
    For MARL Beginners:
    This is an advanced algorithm - start with simpler ones like QMIX or IPPO
    before attempting to understand NFSP.
    """
    
    def __init__(self, agent_id: int, obs_space: Dict, action_space: int, config: Dict):
        """
        Initialize the NFSP agent with all necessary components.
        
        This method sets up:
        1. Basic agent properties (ID, observation space, action space)
        2. NFSP-specific hyperparameters
        3. Neural networks for Q-learning and policy learning
        4. Experience replay buffers for both learning components
        5. Optimizers for training the networks
        
        Args:
            agent_id (int): Unique identifier for this agent (0, 1, 2, ...)
            obs_space (Dict): Observation space defining what the agent can see
                             Example: {'image': (7, 7, 3), 'direction': 4}
            action_space (int): Number of possible actions (e.g., 6 for MultiGrid)
            config (Dict): Configuration dictionary containing hyperparameters
                          Example: {'gamma': 0.99, 'learning_rate': 0.001, ...}
        
        For Beginners:
        Think of this as setting up the agent's "brain" - giving it memory,
        learning algorithms, and the ability to make decisions.
        """
        # Call parent class constructor to set up basic agent properties
        super().__init__(agent_id, obs_space, action_space, config)
        
        # NFSP Core Hyperparameters (these control how the agent learns)
        # Discount factor: how much the agent values future rewards vs immediate rewards
        # gamma = 0.99 means future rewards are worth 99% of immediate rewards
        self.gamma = config.get('gamma', 0.99)
        
        # Exploration parameters: control how much the agent explores vs exploits
        # Start with high exploration (6% random actions) and gradually reduce
        self.epsilon_start = config.get('epsilon_start', 0.06)
        self.epsilon_end = config.get('epsilon_end', 0.001)      # End with 0.1% random actions
        self.epsilon_decay = config.get('epsilon_decay', 0.995)  # Decay rate per episode
        self.epsilon = self.epsilon_start  # Current exploration rate
        
        # NFSP-Specific Parameters (what makes this algorithm unique)
        # eta: Probability of playing best response vs average strategy
        # Higher eta = more exploitation, lower eta = more exploration of average behavior
        self.eta = config.get('eta', 0.1)  # 10% chance of best response, 90% average strategy
        
        # anticipatory_param: How much to anticipate opponent adaptation
        # This is an advanced concept - controls learning rate for average strategy
        self.anticipatory_param = config.get('anticipatory_param', 0.1)
        
        # Neural Networks Setup
        # Q-network: Learns optimal actions for best response (like DQN)
        self.q_network = NFSPQNetwork(obs_space, action_space, config).to(self.device)
        # Target network: Stable version of Q-network for training stability
        self.target_q_network = NFSPQNetwork(obs_space, action_space, config).to(self.device)
        # Copy the Q-network weights to the target network initially
        # Target networks provide stability during training by keeping fixed parameters
        # for computing target Q-values while the main network is being updated
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        
        # Policy network: Learns average strategy through supervised learning
        # This network learns to mimic the historical average behavior of the agent
        self.average_strategy_network = NFSPStrategyNetwork(obs_space, action_space, config).to(self.device)
        
        # Optimizers: Algorithms that update network weights based on gradients
        # Learning rates control how big steps the networks take when learning
        self.q_optimizer = Adam(self.q_network.parameters(), 
                               lr=config.get('q_lr', 1e-3))          # Q-network learning rate
        self.strategy_optimizer = Adam(self.average_strategy_network.parameters(), 
                                     lr=config.get('strategy_lr', 1e-3))  # Strategy network learning rate
        
        # Experience Replay Buffers: Store past experiences for learning
        # RL buffer: Stores (state, action, reward, next_state) for Q-learning
        # SL buffer: Stores (state, action) pairs for supervised learning of average strategy
        self.rl_memory_size = config.get('rl_memory_size', 100000)    # Q-learning buffer size
        self.sl_memory_size = config.get('sl_memory_size', 1000000)   # Strategy learning buffer size
        self.rl_buffer = deque(maxlen=self.rl_memory_size)  # Automatically removes old experiences
        self.sl_buffer = deque(maxlen=self.sl_memory_size)  # Automatically removes old experiences
        
        # Training Control Parameters
        # How often to update the strategy network (every N steps)
        self.strategy_update_freq = config.get('strategy_update_freq', 128)
        self.step_count = 0  # Track number of steps for update scheduling
        
        # Print initialization information for debugging and monitoring
        print(f"Initialized NFSP Agent {agent_id}")
        print(f"Q-network parameters: {sum(p.numel() for p in self.q_network.parameters())}")
        print(f"Strategy network parameters: {sum(p.numel() for p in self.average_strategy_network.parameters())}")
    
    def get_action(self, observation: Dict, mode: str = 'average', training: bool = True) -> Tuple[int, float]:
        """
        Select an action using either best response or average strategy.
        
        This is the core decision-making function. NFSP agents can operate in two modes:
        1. Best Response: Use Q-network to find optimal action against current opponent
        2. Average Strategy: Use policy network to sample from learned average behavior
        
        The choice between modes is controlled by the eta parameter during training.
        
        Args:
            observation (Dict): Current state observation from environment
                               Example: {'image': grid_state, 'direction': facing_direction}
            mode (str): 'best_response' uses Q-network, 'average' uses policy network
            training (bool): Whether agent is in training mode (affects exploration)
            
        Returns:
            Tuple[int, float]: (action_id, action_value)
                              action_id: Integer representing chosen action (0-5 for MultiGrid)
                              action_value: Confidence/value of the action
        
        For Beginners:
        This function is like the agent's "brain" deciding what to do next.
        It can either try to win (best response) or play like it usually does (average).
        """
        # Convert observation dictionary to tensor format for neural networks
        obs_tensor = self._process_observation(observation)
        
        # Disable gradient computation for action selection (saves memory and computation)
        with torch.no_grad():
            if mode == 'best_response':
                # Best Response Mode: Try to find optimal action using Q-network
                # This is like asking "What's the best move against this opponent?"
                
                # Epsilon-greedy exploration: sometimes take random actions to explore
                if training and random.random() < self.epsilon:
                    # Random exploration: choose a random action to discover new strategies
                    # This helps the agent learn about different possibilities
                    action = random.randint(0, self.action_space - 1)
                    action_value = 0.0  # Random actions have no inherent value
                else:
                    # Exploitation: use Q-network to find the best action
                    # Forward pass through neural network to get Q-values for all actions
                    q_values = self.q_network(obs_tensor)[0]  # Get Q-values for all actions
                    # Choose action with highest Q-value (greedy action)
                    action = torch.argmax(q_values).item()
                    # Record the value of the chosen action for analysis
                    action_value = q_values[action].item()
            else:
                # Average Strategy Mode: Use learned average behavior
                # This is like asking "How do I usually play?"
                
                # Get action probabilities from the strategy network
                action_probs = self.average_strategy_network(obs_tensor)[0]
                
                if training:
                    # During training: sample from the probability distribution
                    # This maintains stochasticity in the average strategy
                    action_dist = torch.distributions.Categorical(action_probs)
                    action = action_dist.sample().item()
                else:
                    # During evaluation: choose the most likely action (greedy)
                    action = torch.argmax(action_probs).item()
                
                # Record the probability of the chosen action
                action_value = action_probs[action].item()
        
        return action, action_value
    
    def select_action_mode(self) -> str:
        """
        Select between best response and average strategy modes.
        
        This is a key NFSP concept: the agent randomly chooses whether to:
        1. Play best response (try to win against opponent)
        2. Play average strategy (maintain behavioral diversity)
        
        The eta parameter controls this choice:
        - High eta: More best response (more exploitation)
        - Low eta: More average strategy (more exploration)
        
        Returns:
            str: Either 'best_response' or 'average'
            
        For Beginners:
        Think of this as deciding whether to "play to win" or "play normally".
        This balance is crucial for discovering Nash equilibria.
        """
        return 'best_response' if random.random() < self.eta else 'average'
    
    def add_rl_transition(self, transition: Dict):
        """
        Add a transition to the reinforcement learning buffer.
        
        This stores experiences for training the Q-network (best response learning).
        Each transition contains: (state, action, reward, next_state, done)
        
        Args:
            transition (Dict): Contains 'observation', 'action', 'reward', 
                              'next_observation', 'done'
                              
        For Beginners:
        This is like saving a memory of what happened after taking an action,
        so the agent can learn from it later.
        """
        self.rl_buffer.append(transition)
    
    def add_sl_transition(self, observation: Dict, action: int):
        """
        Add a state-action pair to supervised learning buffer.
        
        This stores experiences for training the average strategy network.
        Unlike RL transitions, these only need state and action (no rewards).
        
        Args:
            observation (Dict): Current state observation
            action (int): Action taken in this state
            
        For Beginners:
        This is like keeping a record of "what I usually do in this situation"
        to learn the average behavior pattern.
        """
        obs_tensor = self._process_observation(observation)
        self.sl_buffer.append({
            'observation': obs_tensor.squeeze(0),  # Remove batch dimension
            'action': action
        })
    
    def update_q_network(self, batch_size: int = 32) -> float:
        """
        Update the Q-network using Deep Q-Network (DQN) learning.
        
        This implements the core Q-learning update:
        Q(s,a) = Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
        
        Steps:
        1. Sample a batch of transitions from replay buffer
        2. Compute current Q-values for taken actions
        3. Compute target Q-values using target network
        4. Minimize the difference (temporal difference error)
        
        Args:
            batch_size (int): Number of transitions to sample for update
            
        Returns:
            float: Training loss value for monitoring
            
        For Beginners:
        This is how the agent learns "how good is each action in each state".
        It compares its current estimates with actual outcomes and adjusts.
        """
        # Check if we have enough experiences to train
        if len(self.rl_buffer) < batch_size:
            return 0.0  # Not enough data yet
        
        # Sample a random batch of transitions from memory
        # Random sampling breaks correlations between consecutive experiences
        batch = random.sample(self.rl_buffer, batch_size)
        
        # Convert batch to tensors for neural network processing
        # Stack creates a batch tensor from individual experience tensors
        observations = torch.stack([t['observation'] for t in batch]).to(self.device)
        actions = torch.tensor([t['action'] for t in batch], dtype=torch.long).to(self.device)
        rewards = torch.tensor([t['reward'] for t in batch], dtype=torch.float32).to(self.device)
        next_observations = torch.stack([t['next_observation'] for t in batch]).to(self.device)
        dones = torch.tensor([t['done'] for t in batch], dtype=torch.bool).to(self.device)
        
        # Compute current Q-values for the actions that were taken
        # gather() selects Q-values for specific actions from the Q-value vector
        current_q_values = self.q_network(observations).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q-values using the target network (for stability)
        with torch.no_grad():  # Don't compute gradients for target calculation
            # Find maximum Q-values for next states
            next_q_values = self.target_q_network(next_observations).max(1)[0]
            # Bellman equation: r + γ * max Q(s',a') if not terminal, else just r
            target_q_values = rewards + (1 - dones.float()) * self.gamma * next_q_values
        
        # Compute loss: Mean Squared Error between current and target Q-values
        q_loss = F.mse_loss(current_q_values, target_q_values)
        
        # Perform gradient descent to minimize loss
        self.q_optimizer.zero_grad()  # Clear previous gradients
        q_loss.backward()             # Compute gradients
        self.q_optimizer.step()       # Update network weights
        
        return q_loss.item()  # Return loss value for monitoring
    
    def update_strategy_network(self, batch_size: int = 128) -> float:
        """
        Update average strategy network using supervised learning.
        
        This trains the policy network to imitate the historical average behavior
        by minimizing the cross-entropy loss between predicted and actual actions.
        
        The goal is to learn P(action|state) for the average strategy.
        
        Args:
            batch_size (int): Number of state-action pairs to sample
            
        Returns:
            float: Training loss value for monitoring
            
        For Beginners:
        This is like learning "what do I typically do in each situation".
        The network learns to predict which actions the agent usually takes.
        """
        # Check if we have enough experiences to train
        if len(self.sl_buffer) < batch_size:
            return 0.0  # Not enough data yet
        
        # Sample a random batch of state-action pairs from supervised learning buffer
        batch = random.sample(self.sl_buffer, batch_size)
        
        # Convert batch to tensors for neural network processing
        observations = torch.stack([t['observation'] for t in batch]).to(self.device)
        # Convert actions to tensor format for loss computation
        actions = torch.tensor([t['action'] for t in batch], dtype=torch.long).to(self.device)
        
        # Forward pass: get action probabilities from strategy network
        action_probs = self.average_strategy_network(observations)
        
        # Compute cross-entropy loss between predicted probabilities and actual actions
        # This is standard supervised learning: minimize difference between prediction and label
        strategy_loss = F.cross_entropy(action_probs, actions)
        
        # Perform gradient descent to minimize loss
        self.strategy_optimizer.zero_grad()  # Clear previous gradients
        strategy_loss.backward()             # Compute gradients
        self.strategy_optimizer.step()       # Update network weights
        
        return strategy_loss.item()  # Return loss value for monitoring
    
    def update_target_network(self):
        """
        Update target Q-network by copying weights from main Q-network.
        
        Target networks are a key stabilization technique in DQN-based algorithms.
        They provide stable target values during training by keeping fixed parameters
        for computing target Q-values while the main network is being updated.
        
        For Beginners:
        Think of this as taking a "snapshot" of your current knowledge to use as
        a reference point. This prevents the agent from "chasing its own tail"
        during learning.
        """
        # Copy all parameters from main Q-network to target Q-network
        self.target_q_network.load_state_dict(self.q_network.state_dict())
    
    def update_exploration(self):
        """
        Decay the exploration rate (epsilon) over time.
        
        As the agent learns more, it should explore less and exploit more.
        This implements epsilon decay: gradually reduce random exploration.
        
        For Beginners:
        Early in training: High exploration (lots of random actions to learn)
        Later in training: Low exploration (use learned knowledge)
        """
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
    
    def _process_observation(self, observation: Dict) -> torch.Tensor:
        """
        Convert observation dictionary to tensor format for neural networks.
        
        This handles the conversion from environment observations (which might be
        images, vectors, or mixed) to the tensor format expected by neural networks.
        
        Args:
            observation (Dict): Raw observation from environment
            
        Returns:
            torch.Tensor: Processed observation ready for neural network
            
        For Beginners:
        Neural networks need numerical tensors as input. This function converts
        whatever the environment gives us (images, numbers, etc.) into the right format.
        """
        # Handle different observation types (implementation depends on environment)
        if 'image' in observation:
            # Convert image observation to tensor
            obs = torch.FloatTensor(observation['image']).unsqueeze(0)
        else:
            # Convert vector observation to tensor
            obs = torch.FloatTensor(observation).unsqueeze(0)
        
        return obs.to(self.device)


class NFSPQNetwork(nn.Module):
    """
    Neural network for Q-value estimation in NFSP.
    
    This network learns Q(state, action) values, which represent the expected
    cumulative reward for taking a specific action in a specific state.
    
    Architecture:
    - Convolutional layers for image processing (if needed)
    - Fully connected layers for decision making
    - Output layer with one value per action
    
    For Beginners:
    Think of this as the "critic" that evaluates how good each action is.
    It learns through trial and error which actions lead to good outcomes.
    """
    
    def __init__(self, obs_space: Dict, action_space: int, config: Dict):
        """
        Initialize the Q-network architecture.
        
        Args:
            obs_space (Dict): Observation space specification
            action_space (int): Number of possible actions
            config (Dict): Network configuration parameters
        """
        super(NFSPQNetwork, self).__init__()
        
        # Network hyperparameters
        self.hidden_size = config.get('hidden_size', 512)  # Size of hidden layers
        
        # Determine input size based on observation space
        if 'image' in obs_space:
            # Convolutional layers for image observations (like MultiGrid)
            # These layers extract spatial features from grid-based observations
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)  # First conv layer
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) # Second conv layer
            self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1) # Third conv layer
            
            # Calculate size after convolutions (depends on input image size)
            conv_out_size = self._get_conv_output_size(obs_space['image'])
            self.fc1 = nn.Linear(conv_out_size, self.hidden_size)
        else:
            # Fully connected layers for vector observations
            input_size = sum(obs_space.values()) if isinstance(obs_space, dict) else obs_space
            self.fc1 = nn.Linear(input_size, self.hidden_size)
        
        # Hidden layers for complex decision making
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, self.hidden_size)
        
        # Output layer: one Q-value per action
        self.q_head = nn.Linear(self.hidden_size, action_space)
        
        # Dropout for regularization (prevents overfitting)
        self.dropout = nn.Dropout(0.1)
    
    def _get_conv_output_size(self, input_shape: Tuple) -> int:
        """
        Calculate the output size after convolutional layers.
        
        This is a utility function to determine how many features we get
        after applying all convolutional layers to the input image.
        
        Args:
            input_shape (Tuple): Shape of input image (height, width, channels)
            
        Returns:
            int: Number of features after convolutions
        """
        # Create dummy input to calculate output size
        dummy_input = torch.zeros(1, 3, input_shape[0], input_shape[1])
        
        # Pass through conv layers
        x = F.relu(self.conv1(dummy_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten and return size
        return x.view(1, -1).size(1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Q-network.
        
        This defines how information flows through the network:
        Input observation → Feature extraction → Decision making → Q-values
        
        Args:
            x (torch.Tensor): Input observation tensor
            
        Returns:
            torch.Tensor: Q-values for each action
            
        For Beginners:
        This is the "thinking" process of the network. It takes what the agent
        sees and outputs how good it thinks each action would be.
        """
        # Handle different input types
        if len(x.shape) == 4:  # Image input (batch, channels, height, width)
            # Apply convolutional layers with ReLU activation
            # ReLU (Rectified Linear Unit) helps the network learn non-linear patterns
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            
            # Flatten convolution output for fully connected layers
            x = x.view(x.size(0), -1)
        
        # Apply fully connected layers with ReLU activation and dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)      # Randomly set some neurons to 0 to prevent overfitting
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        
        # Output Q-values (no activation function - can be positive or negative)
        q_values = self.q_head(x)
        
        return q_values


class NFSPStrategyNetwork(nn.Module):
    """
    Neural network for learning average strategy in NFSP.
    
    This network learns the average policy π(action|state) over all historical
    strategies. It's trained using supervised learning to imitate past behavior.
    
    Key Differences from Q-Network:
    - Outputs action probabilities (not Q-values)
    - Trained with supervised learning (not reinforcement learning)
    - Learns "what I usually do" instead of "what's optimal"
    
    For Beginners:
    Think of this as learning your "typical playing style". It doesn't try to
    find the best moves, just learn what you normally do in each situation.
    """
    
    def __init__(self, obs_space: Dict, action_space: int, config: Dict):
        """
        Initialize the strategy network architecture.
        
        Args:
            obs_space (Dict): Observation space specification
            action_space (int): Number of possible actions
            config (Dict): Network configuration parameters
        """
        super(NFSPStrategyNetwork, self).__init__()
        
        # Network hyperparameters (usually smaller than Q-network)
        self.hidden_size = config.get('strategy_hidden_size', 256)
        
        # Determine input size based on observation space
        if 'image' in obs_space:
            # Convolutional layers for image observations
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
            
            # Calculate size after convolutions
            conv_out_size = self._get_conv_output_size(obs_space['image'])
            self.fc1 = nn.Linear(conv_out_size, self.hidden_size)
        else:
            # Fully connected layers for vector observations
            input_size = sum(obs_space.values()) if isinstance(obs_space, dict) else obs_space
            self.fc1 = nn.Linear(input_size, self.hidden_size)
        
        # Hidden layers for learning behavioral patterns
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        
        # Output layer: probability distribution over actions
        self.policy_head = nn.Linear(self.hidden_size, action_space)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
    
    def _get_conv_output_size(self, input_shape: Tuple) -> int:
        """Calculate output size after convolutional layers."""
        dummy_input = torch.zeros(1, 3, input_shape[0], input_shape[1])
        x = F.relu(self.conv1(dummy_input))
        x = F.relu(self.conv2(x))
        return x.view(1, -1).size(1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the strategy network.
        
        This outputs a probability distribution over actions, representing
        the likelihood of taking each action in the given state.
        
        Args:
            x (torch.Tensor): Input observation tensor
            
        Returns:
            torch.Tensor: Action probabilities (sum to 1)
            
        For Beginners:
        Unlike Q-values, this outputs probabilities. For example:
        [0.6, 0.2, 0.1, 0.05, 0.03, 0.02] means 60% chance of action 0, etc.
        """
        # Handle different input types
        if len(x.shape) == 4:  # Image input
            # Apply convolutional layers
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            
            # Flatten for fully connected layers
            x = x.view(x.size(0), -1)
        
        # Apply fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        
        # Output action probabilities using softmax
        # Softmax ensures all probabilities sum to 1
        action_probs = F.softmax(self.policy_head(x), dim=-1)
        
        return action_probs


class NFSP(MARLAlgorithm):
    """
    Neural Fictitious Self-Play (NFSP) Multi-Agent Algorithm.
    
    This class coordinates multiple NFSP agents and handles the training process.
    It implements the multi-agent version of NFSP for competitive scenarios.
    
    Key Features:
    1. Self-play training between agents
    2. Dual learning objectives (best response + average strategy)
    3. Nash equilibrium approximation through fictitious play
    4. Automatic opponent adaptation
    
    For Beginners:
    This is the "trainer" that manages multiple NFSP agents. It makes them
    play against each other and coordinates their learning.
    """
    
    def __init__(self, config: Dict, env_info: Dict):
        """
        Initialize the NFSP algorithm with multiple agents.
        
        Args:
            config (Dict): Algorithm configuration
            env_info (Dict): Environment information including observation and action spaces
        """
        super().__init__(config, env_info)
        
        # Create NFSP agents for each position in the environment
        self.agents = {}
        for agent_id in range(env_info['num_agents']):
            self.agents[agent_id] = NFSPAgent(
                agent_id=agent_id,
                obs_space=env_info['obs_space'],
                action_space=env_info['action_space'],
                config=config
            )
        
        # Training configuration
        self.target_update_freq = config.get('target_update_freq', 1000)  # How often to update target networks
        self.training_freq = config.get('training_freq', 4)               # How often to train networks
        self.strategy_update_freq = config.get('strategy_update_freq', 128)  # Strategy network update frequency
        
        # Episode tracking
        self.episode_count = 0
        self.step_count = 0
        
        print(f"Initialized NFSP with {len(self.agents)} agents")
        print(f"Target update frequency: {self.target_update_freq}")
        print(f"Training frequency: {self.training_freq}")
    
    def get_actions(self, observations: Dict, training: bool = True) -> Dict:
        """
        Get actions from all agents for the current step.
        
        This is where the NFSP strategy selection happens:
        1. Each agent chooses between best response and average strategy
        2. Agents select actions based on their chosen mode
        3. Actions are stored for supervised learning
        
        Args:
            observations (Dict): Observations for all agents
            training (bool): Whether in training mode
            
        Returns:
            Dict: Actions for all agents
            
        For Beginners:
        This is like asking each agent "what do you want to do?" and
        collecting all their decisions.
        """
        actions = {}
        action_values = {}
        
        for agent_id, agent in self.agents.items():
            # Each agent selects its mode (best response vs average strategy)
            mode = agent.select_action_mode() if training else 'best_response'
            
            # Get action based on selected mode
            action, action_value = agent.get_action(
                observation=observations[agent_id],
                mode=mode,
                training=training
            )
            
            actions[agent_id] = action
            action_values[agent_id] = action_value
            
            # Store state-action pairs for supervised learning (average strategy)
            if training and mode == 'average':
                agent.add_sl_transition(observations[agent_id], action)
        
        # Store action selection information for analysis
        self.last_action_info = {
            'actions': actions,
            'action_values': action_values
        }
        
        return actions
    
    def update(self, batch_experiences: List[Dict]) -> Dict:
        """
        Update all agents based on collected experiences.
        
        This performs the core NFSP learning:
        1. Add experiences to RL buffers for Q-learning
        2. Train Q-networks (best response learning)
        3. Train strategy networks (average behavior learning)
        4. Update target networks periodically
        5. Decay exploration rates
        
        Args:
            batch_experiences (List[Dict]): List of experience transitions
            
        Returns:
            Dict: Training statistics and losses
            
        For Beginners:
        This is the "learning" phase where agents reflect on what happened
        and update their knowledge.
        """
        # Organize experiences by agent
        agent_experiences = {agent_id: [] for agent_id in self.agents.keys()}
        
        for experience in batch_experiences:
            for agent_id in self.agents.keys():
                if agent_id in experience:
                    agent_experiences[agent_id].append(experience[agent_id])
        
        # Training statistics
        training_stats = {
            'q_losses': {},
            'strategy_losses': {},
            'epsilon_values': {}
        }
        
        # Update each agent
        for agent_id, agent in self.agents.items():
            # Add experiences to RL buffer
            for exp in agent_experiences[agent_id]:
                agent.add_rl_transition(exp)
            
            # Train networks if it's time
            if self.step_count % self.training_freq == 0:
                # Update Q-network (best response learning)
                q_loss = agent.update_q_network()
                training_stats['q_losses'][agent_id] = q_loss
                
                # Update strategy network (average behavior learning)
                if self.step_count % self.strategy_update_freq == 0:
                    strategy_loss = agent.update_strategy_network()
                    training_stats['strategy_losses'][agent_id] = strategy_loss
            
            # Update target networks periodically
            if self.step_count % self.target_update_freq == 0:
                agent.update_target_network()
            
            # Decay exploration
            agent.update_exploration()
            training_stats['epsilon_values'][agent_id] = agent.epsilon
        
        self.step_count += 1
        
        return training_stats
    
    def get_algorithm_info(self) -> Dict:
        """
        Get information about the current state of the algorithm.
        
        Returns:
            Dict: Algorithm state information
        """
        return {
            'algorithm': 'NFSP',
            'episode_count': self.episode_count,
            'step_count': self.step_count,
            'num_agents': len(self.agents),
            'agent_epsilons': {aid: agent.epsilon for aid, agent in self.agents.items()},
            'buffer_sizes': {
                aid: {
                    'rl_buffer': len(agent.rl_buffer),
                    'sl_buffer': len(agent.sl_buffer)
                } for aid, agent in self.agents.items()
            }
        }
            'q_network_state_dict': self.q_network.state_dict(),
            'target_q_network_state_dict': self.target_q_network.state_dict(),
            'average_strategy_network_state_dict': self.average_strategy_network.state_dict(),
            'q_optimizer_state_dict': self.q_optimizer.state_dict(),
            'strategy_optimizer_state_dict': self.strategy_optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'agent_id': self.agent_id,
            'config': self.config
        }
        torch.save(checkpoint, f"{path}_nfsp_agent_{self.agent_id}.pth")
        print(f"Saved NFSP Agent {self.agent_id} model to {path}_nfsp_agent_{self.agent_id}.pth")
    
    def load_model(self, path: str):
        """Load the agent's model."""
        checkpoint = torch.load(f"{path}_nfsp_agent_{self.agent_id}.pth", map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_q_network.load_state_dict(checkpoint['target_q_network_state_dict'])
        self.average_strategy_network.load_state_dict(checkpoint['average_strategy_network_state_dict'])
        self.q_optimizer.load_state_dict(checkpoint['q_optimizer_state_dict'])
        self.strategy_optimizer.load_state_dict(checkpoint['strategy_optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.step_count = checkpoint['step_count']
        
        print(f"Loaded NFSP Agent {self.agent_id} model from {path}_nfsp_agent_{self.agent_id}.pth")


class NFSPQNetwork(nn.Module):
    """Q-network for NFSP best response."""
    
    def __init__(self, obs_space: Dict, action_space: int, config: Dict):
        super().__init__()
        
        self.obs_dim = self._estimate_obs_dim(obs_space)
        hidden_dim = config.get('hidden_dim', 256)
        
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
        """Forward pass through Q-network."""
        return self.network(obs)


class NFSPStrategyNetwork(nn.Module):
    """Strategy network for NFSP average strategy."""
    
    def __init__(self, obs_space: Dict, action_space: int, config: Dict):
        super().__init__()
        
        self.obs_dim = self._estimate_obs_dim(obs_space)
        hidden_dim = config.get('hidden_dim', 256)
        
        self.network = nn.Sequential(
            nn.Linear(self.obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_space),
            nn.Softmax(dim=-1)
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
        """Forward pass through strategy network."""
        return self.network(obs)


class NFSP(MARLAlgorithm):
    """
    Neural Fictitious Self-Play (NFSP) algorithm.
    
    Combines reinforcement learning with supervised learning to approximate
    Nash equilibria in multi-agent games.
    """
    
    def __init__(self, env, config: Dict, device: torch.device):
        """
        Initialize NFSP algorithm.
        
        Args:
            env: Multi-agent environment
            config: Configuration dictionary
            device: PyTorch device
        """
        super().__init__(env, config, device)
        
        # NFSP specific parameters
        self.train_start = config.get('train_start', 1000)
        self.update_interval = config.get('update_interval', 1)
        self.target_update_freq = config.get('target_update_freq', 1000)
        self.strategy_update_freq = config.get('strategy_update_freq', 128)
        
        print(f"Initialized NFSP")
        print(f"Number of agents: {self.n_agents}")
        print(f"Target update frequency: {self.target_update_freq}")
        print(f"Strategy update frequency: {self.strategy_update_freq}")
    
    def _create_agents(self):
        """Create NFSP agents."""
        obs_space = self._get_obs_space()
        action_space = len(self.env.actions)
        
        for i in range(self.n_agents):
            agent = NFSPAgent(i, obs_space, action_space, self.config)
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
        """Collect rollout with NFSP mode selection."""
        obs = env.reset()
        done = False
        step_count = 0
        total_reward = [0.0] * self.n_agents
        
        # Select action modes for each agent
        agent_modes = [agent.select_action_mode() for agent in self.agents]
        
        while not done:
            # Get actions from all agents
            actions = []
            for i, agent in enumerate(self.agents):
                agent_obs = self._extract_agent_obs(obs, i)
                action, _ = agent.get_action(agent_obs, mode=agent_modes[i], training=True)
                actions.append(action)
                
                # Add to SL buffer if using best response
                if agent_modes[i] == 'best_response':
                    agent.add_sl_transition(agent_obs, action)
            
            # Take environment step
            next_obs, rewards, done, info = env.step(actions)
            
            # Store RL transitions
            for i in range(self.n_agents):
                if agent_modes[i] == 'best_response':
                    agent_obs = self._extract_agent_obs(obs, i)
                    next_agent_obs = self._extract_agent_obs(next_obs, i)
                    
                    reward = rewards[i] if isinstance(rewards, list) else rewards
                    total_reward[i] += reward
                    
                    obs_tensor = self.agents[i]._process_observation(agent_obs)
                    next_obs_tensor = self.agents[i]._process_observation(next_agent_obs)
                    
                    transition = {
                        'observation': obs_tensor.squeeze(0),
                        'action': actions[i],
                        'reward': reward,
                        'next_observation': next_obs_tensor.squeeze(0),
                        'done': done
                    }
                    
                    self.agents[i].add_rl_transition(transition)
            
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
        """Perform NFSP training step."""
        if self.total_steps < self.train_start:
            return {
                'episode_length': rollout_data['episode_length'],
                'total_reward': sum(rollout_data['total_reward']) / self.n_agents
            }
        
        # Train each agent
        if self.total_steps % self.update_interval == 0:
            q_losses = []
            strategy_losses = []
            
            for agent in self.agents:
                # Update Q-network
                q_loss = agent.update_q_network()
                q_losses.append(q_loss)
                
                # Update strategy network
                if self.total_steps % self.strategy_update_freq == 0:
                    strategy_loss = agent.update_strategy_network()
                    strategy_losses.append(strategy_loss)
            
            # Update target networks
            if self.total_steps % self.target_update_freq == 0:
                for agent in self.agents:
                    agent.update_target_network()
                print(f"Updated target networks at step {self.total_steps}")
            
            metrics = {
                'avg_q_loss': np.mean(q_losses) if q_losses else 0.0,
                'episode_length': rollout_data['episode_length'],
                'total_reward': sum(rollout_data['total_reward']) / self.n_agents,
                'total_steps': self.total_steps
            }
            
            if strategy_losses:
                metrics['avg_strategy_loss'] = np.mean(strategy_losses)
            
            return metrics
        
        return {
            'episode_length': rollout_data['episode_length'],
            'total_reward': sum(rollout_data['total_reward']) / self.n_agents,
            'total_steps': self.total_steps
        }
