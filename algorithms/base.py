"""
(C) Shreyan Mitra, based on starter code by Natasha Jaques

Base Classes for Multi-Agent Reinforcement Learning (MARL) Algorithms

This module provides the fundamental building blocks for implementing MARL algorithms.
Think of this as the "blueprint" or "template" that all MARL algorithms follow.

Key Concepts for Beginners:
- Abstract Base Classes: Templates that define what methods all algorithms must have
- Inheritance: New algorithms inherit common functionality from these base classes
- Modularity: Each algorithm can focus on its unique logic while sharing common code
- Type Safety: Clear interfaces ensure all algorithms work together consistently

What You'll Find Here:
1. MARLAgent: Base class for individual learning agents
2. MARLAlgorithm: Base class for multi-agent coordination algorithms
3. Common utilities and helper functions
4. Standard interfaces for training and evaluation

For MARL Beginners:
Start by understanding these base classes before diving into specific algorithms.
Every algorithm in this framework (QMIX, MADDPG, IPPO, etc.) builds on these foundations.
"""

# Import essential libraries for abstract classes and type checking
from abc import ABC, abstractmethod  # ABC = Abstract Base Class for creating templates
import numpy as np                   # Numerical computations and array operations
from typing import Dict, List, Tuple, Any, Optional, Union  # Type hints for code clarity

# Try to import PyTorch (deep learning framework)
# This is optional - some simple algorithms might not need neural networks
try:
    import torch                     # Main PyTorch library
    import torch.nn as nn           # Neural network modules
    import torch.nn.functional as F # Activation functions and utilities
    from torch.distributions import Categorical  # Probability distributions
    TORCH_AVAILABLE = True
except ImportError:
    # If PyTorch isn't installed, provide helpful message
    print("Warning: PyTorch not available. Install with: pip install torch")
    TORCH_AVAILABLE = False


class MARLAgent(ABC):
    """
    Abstract Base Class for Multi-Agent Reinforcement Learning Agents.
    
    This is the fundamental template that ALL individual agents must follow,
    regardless of which algorithm they implement (QMIX, MADDPG, IPPO, etc.).
    
    Think of this as defining "what it means to be a learning agent":
    1. Agents must be able to select actions given observations
    2. Agents must be able to learn from experiences
    3. Agents must be able to save/load their learned knowledge
    4. Agents must have unique identities and understand their environment
    
    Key Responsibilities:
    - Action Selection: Choose what to do in each situation
    - Learning: Update knowledge based on outcomes
    - State Management: Keep track of internal learning state
    - Persistence: Save and load learned policies
    
    For MARL Beginners:
    Every agent is like a student learning to play a game. This class defines
    what every student must be able to do, but each algorithm teaches differently.
    """
    
    def __init__(self, agent_id: int, obs_space: Dict, action_space: int, config: Dict):
        """
        Initialize the MARL agent with essential properties.
        
        This constructor sets up the basic properties that every agent needs,
        regardless of their learning algorithm. Think of this as enrolling
        a student in school - giving them an ID, telling them what they can
        see and do, and providing learning guidelines.
        
        Args:
            agent_id (int): Unique identifier for this agent (like a student ID)
                           Example: 0, 1, 2 for a 3-agent system
            obs_space (Dict): What the agent can observe from the environment
                             Example: {'image': (7, 7, 3), 'direction': 4}
                             This means agent sees 7x7 RGB image + direction
            action_space (int): Number of possible actions agent can take
                               Example: 6 for MultiGrid (up, down, left, right, toggle, done)
            config (Dict): Learning configuration and hyperparameters
                          Example: {'learning_rate': 0.001, 'gamma': 0.99, ...}
        
        For Beginners:
        This is like setting up the agent's "character sheet" with their ID,
        what they can see, what they can do, and how they should learn.
        """
        # Store essential agent properties
        self.agent_id = agent_id      # Unique identifier (which agent am I?)
        self.obs_space = obs_space    # What can I observe? (my "senses")
        self.action_space = action_space  # What can I do? (my "actions")
        self.config = config          # How should I learn? (my "learning rules")
        
        # Set up computing device (GPU if available, otherwise CPU)
        # Neural networks train much faster on GPUs
        if TORCH_AVAILABLE:
            # Check if CUDA (NVIDIA GPU support) is available
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Agent {agent_id} using device: {self.device}")
        else:
            self.device = "cpu"  # Fallback if PyTorch not available
        
        # Initialize memory for storing experiences (like a learning diary)
        # Different algorithms will store different types of experiences here
        self.memory = {}
        self.reset_memory()  # Start with empty memory
    
    @abstractmethod
    def get_action(self, observation: Dict, training: bool = True) -> Tuple[int, float]:
        """
        Select an action given the current observation.
        
        This is the core decision-making method that every agent must implement.
        It's like asking the agent "What do you want to do in this situation?"
        
        Different algorithms will implement this differently:
        - Q-learning: Choose action with highest Q-value
        - Policy gradient: Sample from learned probability distribution
        - Actor-Critic: Use policy network to select action
        
        Args:
            observation (Dict): Current state of the environment as seen by this agent
                               Example: {'image': grid_state, 'direction': facing_dir}
            training (bool): Whether agent is in training mode (affects exploration)
                            True: May explore with random actions
                            False: Use best known policy (no exploration)
            
        Returns:
            Tuple[int, float]: (action_id, action_value)
                              action_id: Integer representing chosen action
                              action_value: Confidence or probability of action
        
        For Beginners:
        This is the agent's "brain" - where it decides what to do next.
        Every algorithm must implement this, but they decide differently.
        """
        pass  # Must be implemented by specific algorithm classes
    
    @abstractmethod
    def update(self, batch_data: Dict) -> Dict[str, float]:
        """
        Update the agent's knowledge based on collected experiences.
        
        This is where the actual "learning" happens. After the agent acts in
        the environment and sees the results, this method updates its policy
        to make better decisions in the future.
        
        Different algorithms learn differently:
        - Q-learning: Update Q-values using Bellman equation
        - Policy gradient: Update policy to increase probability of good actions
        - Actor-Critic: Update both value estimates and policy
        
        Args:
            batch_data (Dict): Collection of experiences to learn from
                              Example: {'observations': [...], 'actions': [...], 
                                       'rewards': [...], 'next_observations': [...]}
            
        Returns:
            Dict[str, float]: Training statistics and loss values
                             Example: {'loss': 0.05, 'value_loss': 0.03, 'policy_loss': 0.02}
        
        For Beginners:
        This is like studying after an exam - the agent looks at what happened
        and adjusts its strategy to do better next time.
        """
        pass  # Must be implemented by specific algorithm classes
    
    @abstractmethod
    def reset_memory(self):
        """
        Reset the agent's memory/experience buffer.
        
        This clears all stored experiences, typically called at the start
        of training or when switching between different training phases.
        
        For Beginners:
        This is like clearing the agent's "learning diary" to start fresh.
        """
        pass  # Must be implemented by specific algorithm classes
    
    @abstractmethod
    def save_model(self, path: str):
        """Save the agent's model to disk."""
        pass
    @abstractmethod
    def save_model(self, path: str):
        """
        Save the agent's learned model to disk.
        
        This allows the agent to persist its learned knowledge so it can
        be loaded later for evaluation or continued training.
        
        Args:
            path (str): File path where to save the model
                       Example: "models/agent_0_qmix.pt"
        
        For Beginners:
        This is like saving your progress in a video game - you can come back
        to it later without losing what you've learned.
        """
        pass  # Must be implemented by specific algorithm classes
    
    @abstractmethod
    def load_model(self, path: str):
        """
        Load the agent's model from disk.
        
        This restores previously learned knowledge from a saved file,
        allowing evaluation or continued training from a checkpoint.
        
        Args:
            path (str): File path where the model is saved
                       Example: "models/agent_0_qmix.pt"
        
        For Beginners:
        This is like loading your saved game progress - the agent remembers
        everything it learned before.
        """
        pass  # Must be implemented by specific algorithm classes


class MARLAlgorithm(ABC):
    """
    Abstract Base Class for Multi-Agent Reinforcement Learning Algorithms.
    
    This class manages the coordination and training of multiple agents working
    together or competing in the same environment. While MARLAgent handles
    individual decision-making, MARLAlgorithm handles the "big picture":
    
    Key Responsibilities:
    1. Agent Management: Create, coordinate, and manage multiple agents
    2. Experience Collection: Gather experiences from environment interactions
    3. Training Coordination: Orchestrate learning across all agents
    4. Communication: Handle information sharing between agents (if needed)
    5. Evaluation: Assess overall system performance
    
    Different MARL Paradigms:
    - Centralized Training, Decentralized Execution (CTDE): Train together, act alone
    - Independent Learning: Each agent learns independently
    - Communication-based: Agents share information during execution
    - Competitive: Agents learn to compete against each other
    
    For MARL Beginners:
    Think of this as the "teacher" or "coach" managing a team of learning agents.
    While each agent (student) learns individually, the algorithm coordinates
    their learning and ensures they work well together.
    """
    
    def __init__(self, env, config: Dict, device):
        """
        Initialize the multi-agent algorithm with environment and configuration.
        
        This sets up the overall learning system that will manage multiple agents
        and coordinate their training process.
        
        Args:
            env: Multi-agent environment where agents will learn
                 Must provide observations, handle actions, and return rewards
            config (Dict): Algorithm configuration and hyperparameters
                          Example: {'batch_size': 32, 'learning_rate': 0.001, ...}
            device: Computing device for neural networks (CPU/GPU)
                   Can be torch.device or string like "cuda" or "cpu"
        
        For Beginners:
        This is like setting up a classroom where multiple students (agents)
        will learn together under the guidance of a teacher (algorithm).
        """
        # Store essential algorithm properties
        self.env = env                    # The environment where agents learn
        self.config = config              # Learning configuration and hyperparameters
        self.device = device              # Computing device for neural networks
        self.n_agents = env.n_agents      # Number of agents in the system
        
        # Create container for all agents
        self.agents = []
        self._create_agents()             # Let specific algorithm create its agents
        
        # Training progress tracking
        self.episode_count = 0            # How many episodes have been completed
        self.total_steps = 0              # Total number of environment steps taken
        
        print(f"Initialized {self.__class__.__name__} with {self.n_agents} agents")
        print(f"Using device: {self.device}")
        
    @abstractmethod
    def _create_agents(self):
        """
        Create and initialize all agents for this algorithm.
        
        This method is called during initialization and must be implemented
        by each specific algorithm to create the appropriate type and number
        of agents.
        
        Different algorithms create different types of agents:
        - QMIX: Create Q-learning agents + mixing network
        - MADDPG: Create actor-critic agents with centralized critics
        - IPPO: Create independent PPO agents
        
        For Beginners:
        This is like enrolling students in a class - each teaching method
        (algorithm) might need different types of students with different
        capabilities.
        """
        pass  # Must be implemented by specific algorithm classes
    
    @abstractmethod
    def collect_rollout(self, env) -> Dict:
        """
        Collect a rollout of experiences from the environment.
        
        A "rollout" is a sequence of interactions where agents act in the
        environment, observe results, and collect experiences for learning.
        This is the data collection phase of reinforcement learning.
        
        Typical rollout process:
        1. Reset environment to starting state
        2. Loop: agents observe → agents act → environment responds
        3. Continue until episode ends or max steps reached
        4. Return collected experiences for training
        
        Args:
            env: The environment to collect experiences from
            
        Returns:
            Dict: Dictionary containing collected experiences
                  Example: {
                      'observations': [...],  # What agents saw
                      'actions': [...],       # What agents did
                      'rewards': [...],       # What agents received
                      'dones': [...]          # When episodes ended
                  }
        
        For Beginners:
        This is like having students practice and take notes on what happens.
        The experiences collected here are what the agents will learn from.
        """
        pass  # Must be implemented by specific algorithm classes
    
    @abstractmethod
    def train_step(self, rollout_data: Dict) -> Dict[str, float]:
        """
        Perform one training step using collected experiences.
        
        This is where the actual learning happens. After collecting experiences
        from environment interactions, this method updates all agents' policies
        to improve their performance.
        
        Different algorithms train differently:
        - Value-based (QMIX): Update Q-functions using temporal difference learning
        - Policy-based (MAPPO): Update policies using policy gradient methods
        - Actor-Critic (MADDPG): Update both value functions and policies
        
        Args:
            rollout_data (Dict): Experiences collected from environment
                                Contains observations, actions, rewards, etc.
            
        Returns:
            Dict[str, float]: Training metrics and loss values
                             Example: {'total_loss': 0.15, 'policy_loss': 0.08, 'value_loss': 0.07}
        
        For Beginners:
        This is the "study session" where agents analyze their experiences
        and update their strategies to perform better next time.
        """
        pass  # Must be implemented by specific algorithm classes
    
    def train(self, total_episodes: int) -> None:
        """
        Main training loop that coordinates the entire learning process.
        
        This is the "master loop" that:
        1. Collects experiences from environment
        2. Updates agent policies based on experiences
        3. Logs progress and saves models periodically
        4. Repeats until training is complete
        
        Args:
            total_episodes (int): Total number of episodes to train for
                                 An episode is one complete run from start to finish
        
        For Beginners:
        Think of this as the overall "curriculum" - it manages the entire
        learning process from start to finish, making sure agents get enough
        practice and their progress is tracked.
        """
        print(f"Starting training for {total_episodes} episodes...")
        
        for episode in range(total_episodes):
            # Phase 1: Practice (collect experiences from environment)
            rollout_data = self.collect_rollout(self.env)
            
            # Phase 2: Study (learn from experiences)
            metrics = self.train_step(rollout_data)
            
            # Phase 3: Progress tracking (log and save periodically)
            if episode % self.config.get('log_interval', 100) == 0:
                self._log_metrics(episode, metrics)
            
            if episode % self.config.get('save_interval', 1000) == 0:
                self.save_models(f"episode_{episode}")
            
            self.episode_count += 1
        
        print(f"Training completed after {total_episodes} episodes!")
    
    def _log_metrics(self, episode: int, metrics: Dict[str, float]):
        """
        Log training progress and metrics.
        
        This prints or saves training statistics so we can monitor
        how well the agents are learning.
        
        Args:
            episode (int): Current episode number
            metrics (Dict[str, float]): Training metrics to log
        
        For Beginners:
        This is like keeping a report card - tracking how well the
        agents are doing and whether they're improving over time.
        """
        print(f"Episode {episode}: {metrics}")
        # Could also log to files, tensorboard, wandb, etc.
    
    def save_models(self, suffix: str = ""):
        """
        Save all trained agent models to disk.
        
        This preserves the learned knowledge so it can be used later
        for evaluation or continued training.
        
        Args:
            suffix (str): Optional suffix for filenames
                         Example: "episode_1000" creates "agent_0_episode_1000.pt"
        
        For Beginners:
        This is like saving your progress in a video game - you can
        come back later without losing what you've learned.
        """
        for i, agent in enumerate(self.agents):
            filename = f"agent_{i}_{suffix}.pt" if suffix else f"agent_{i}.pt"
            agent.save_model(filename)
    
    def load_models(self, suffix: str = ""):
        """
        Load previously saved agent models from disk.
        
        This restores learned knowledge from previous training sessions.
        
        Args:
            suffix (str): Optional suffix for filenames to load
        
        For Beginners:
        This is like loading your saved game progress - the agents
        remember everything they learned before.
        """
        for i, agent in enumerate(self.agents):
            filename = f"agent_{i}_{suffix}.pt" if suffix else f"agent_{i}.pt"
            agent.load_model(filename)


class ReplayBuffer:
    """
    Experience Replay Buffer for Off-Policy Learning Algorithms.
    
    This is a crucial component for many deep reinforcement learning algorithms.
    It stores past experiences and allows agents to learn from them multiple times,
    which improves sample efficiency and training stability.
    
    Key Benefits:
    1. Sample Efficiency: Learn from each experience multiple times
    2. Stability: Break correlations between consecutive experiences
    3. Batch Learning: Train on batches instead of single samples
    
    Used by algorithms like: MADDPG, QMIX, DQN-based methods
    
    For MARL Beginners:
    Think of this as the agents' "memory bank" where they store important
    experiences to study later. Instead of forgetting experiences immediately,
    they keep them for repeated study sessions.
    """
    
    def __init__(self, capacity: int, obs_shape: Tuple, action_dim: int, n_agents: int):
        """
        Initialize the experience replay buffer.
        
        This creates storage arrays for different types of experience data
        that will be collected during training.
        
        Args:
            capacity (int): Maximum number of experiences to store
                           When full, oldest experiences are overwritten
            obs_shape (Tuple): Shape of individual agent observations
                              Example: (7, 7, 3) for 7x7 RGB images
            action_dim (int): Dimension of action space
                             For discrete: 1, for continuous: action vector size
            n_agents (int): Number of agents in the system
        
        For Beginners:
        This is like setting up a filing cabinet with specific compartments
        for different types of information about each experience.
        """
        self.capacity = capacity    # Maximum storage capacity
        self.size = 0              # Current number of stored experiences
        self.ptr = 0               # Pointer to next storage location (circular buffer)
        
        # Pre-allocate arrays for efficient storage (much faster than lists)
        # Each array stores one type of information for all experiences
        
        # What agents observed (their "vision")
        self.observations = np.zeros((capacity, n_agents) + obs_shape, dtype=np.float32)
        
        # What actions agents took (their "decisions")
        self.actions = np.zeros((capacity, n_agents, action_dim), dtype=np.float32)
        
        # What rewards agents received (their "feedback")
        self.rewards = np.zeros((capacity, n_agents), dtype=np.float32)
        
        # What agents observed after taking actions (their "new vision")
        self.next_observations = np.zeros((capacity, n_agents) + obs_shape, dtype=np.float32)
        
        # Whether episodes ended (their "terminal states")
        self.dones = np.zeros((capacity, n_agents), dtype=np.float32)
    
    def add(self, obs: np.ndarray, actions: np.ndarray, rewards: np.ndarray, 
            next_obs: np.ndarray, dones: np.ndarray):
        """
        Add a new experience transition to the buffer.
        
        This stores one complete "experience" - what happened when agents
        took specific actions in specific states.
        
        Args:
            obs (np.ndarray): Observations before taking actions
            actions (np.ndarray): Actions taken by agents
            rewards (np.ndarray): Rewards received after actions
            next_obs (np.ndarray): Observations after taking actions
            dones (np.ndarray): Whether episodes ended after actions
        
        For Beginners:
        This is like writing down what happened: "I was in situation X,
        did action Y, got reward Z, and ended up in situation W."
        """
        # Store the experience at current pointer location
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = actions
        self.rewards[self.ptr] = rewards
        self.next_observations[self.ptr] = next_obs
        self.dones[self.ptr] = dones
        
        # Move pointer to next location (circular: wraps around when full)
        self.ptr = (self.ptr + 1) % self.capacity
        
        # Update size (capped at capacity)
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Dict[str, Any]:
        """
        Sample a random batch of experiences for training.
        
        Random sampling is crucial for breaking correlations between
        consecutive experiences, which improves training stability.
        
        Args:
            batch_size (int): Number of experiences to sample
                             Typical values: 32, 64, 128
            
        Returns:
            Dict[str, Any]: Batch of experiences ready for training
        
        For Beginners:
        This is like randomly picking some experiences from the memory
        to study. Random selection helps avoid getting stuck on patterns
        from recent experiences.
        """
        # Randomly select experience indices
        indices = np.random.choice(self.size, batch_size, replace=False)
        
        # Return a batch dictionary with all experience components
        batch = {
            'observations': self.observations[indices],
            'actions': self.actions[indices],
            'rewards': self.rewards[indices],
            'next_observations': self.next_observations[indices],
            'dones': self.dones[indices]
        }
        
        return batch
    
    def __len__(self):
        """
        Return current number of stored experiences.
        
        Returns:
            int: Number of experiences currently in buffer
        
        For Beginners:
        This lets you check how many experiences are stored, like asking
        "how full is my memory bank?"
        """
        return self.size
    
    def is_ready(self, min_size: int) -> bool:
        """
        Check if buffer has enough experiences for training.
        
        Many algorithms need a minimum number of experiences before
        they can start learning effectively.
        
        Args:
            min_size (int): Minimum number of experiences needed
            
        Returns:
            bool: True if buffer has enough experiences
        
        For Beginners:
        This is like checking "do I have enough study material before
        starting my learning session?"
        """
        return self.size >= min_size


# Utility Functions for MARL Algorithms

def compute_gae(rewards: np.ndarray, values: np.ndarray, next_values: np.ndarray, 
                dones: np.ndarray, gamma: float = 0.99, gae_lambda: float = 0.95) -> np.ndarray:
    """
    Compute Generalized Advantage Estimation (GAE) for policy gradient methods.
    
    GAE is a technique to estimate how much better an action was compared to
    the average action in that state. It's used in algorithms like PPO and A3C.
    
    Args:
        rewards (np.ndarray): Rewards received at each step
        values (np.ndarray): Value estimates for each state
        next_values (np.ndarray): Value estimates for next states
        dones (np.ndarray): Whether episodes ended at each step
        gamma (float): Discount factor for future rewards
        gae_lambda (float): GAE smoothing parameter
        
    Returns:
        np.ndarray: Advantage estimates for each step
    
    For MARL Beginners:
    This is an advanced concept - it estimates "how good was this action
    compared to what I usually do in this situation?" Don't worry about
    the math details initially; just know it helps policy gradient methods
    learn more effectively.
    """
    advantages = np.zeros_like(rewards)
    advantage = 0
    
    # Compute advantages backwards through time
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = next_values[t]
        else:
            next_value = values[t + 1]
        
        # Temporal difference error
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        
        # GAE advantage
        advantage = delta + gamma * gae_lambda * (1 - dones[t]) * advantage
        advantages[t] = advantage
    
    return advantages


def normalize_advantages(advantages: np.ndarray) -> np.ndarray:
    """
    Normalize advantages to have zero mean and unit variance.
    
    This is a common preprocessing step in policy gradient methods
    that helps with training stability and convergence.
    
    Args:
        advantages (np.ndarray): Raw advantage estimates
        
    Returns:
        np.ndarray: Normalized advantages
    
    For Beginners:
    This is like "grading on a curve" - it adjusts the advantage estimates
    to have a standard scale, which helps the learning algorithm work better.
    """
    return (advantages - advantages.mean()) / (advantages.std() + 1e-8)


def soft_update(target_network: nn.Module, source_network: nn.Module, tau: float = 0.001):
    """
    Perform soft update of target network parameters.
    
    This gradually updates a target network by mixing its parameters
    with those of a source network. Used in algorithms like DDPG and TD3.
    
    Args:
        target_network (nn.Module): Network to be updated
        source_network (nn.Module): Network to copy from
        tau (float): Update rate (0 = no update, 1 = full copy)
    
    For Beginners:
    This is like gradually updating your "reference book" with new knowledge
    instead of completely replacing it all at once. The slow update helps
    keep training stable.
    """
    if not TORCH_AVAILABLE:
        return
        
    for target_param, source_param in zip(target_network.parameters(), source_network.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)
        
        if TORCH_AVAILABLE:
            # Convert to tensors if PyTorch is available
            for key in batch:
                batch[key] = torch.FloatTensor(batch[key])
        
        return batch
    
    def __len__(self) -> int:
        return self.size


def compute_gae(rewards, values, next_values, dones, gamma: float = 0.99, lambda_: float = 0.95):
    """
    Compute Generalized Advantage Estimation (GAE).
    
    Args:
        rewards: Tensor of rewards [T, N]
        values: Tensor of value estimates [T, N]
        next_values: Tensor of next value estimates [T, N]
        dones: Tensor of done flags [T, N]
        gamma: Discount factor
        lambda_: GAE lambda parameter
        
    Returns:
        Tensor of GAE advantages [T, N]
    """
    if not TORCH_AVAILABLE:
        return np.zeros_like(rewards)
    
    advantages = torch.zeros_like(rewards)
    last_advantage = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = next_values[t]
        else:
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        advantages[t] = delta + gamma * lambda_ * (1 - dones[t]) * last_advantage
        last_advantage = advantages[t]
    
    return advantages


def compute_returns(rewards, values, dones, gamma: float = 0.99):
    """
    Compute discounted returns.
    
    Args:
        rewards: Tensor of rewards [T, N]
        values: Tensor of value estimates [T, N] 
        dones: Tensor of done flags [T, N]
        gamma: Discount factor
        
    Returns:
        Tensor of returns [T, N]
    """
    if not TORCH_AVAILABLE:
        return np.zeros_like(rewards)
    
    returns = torch.zeros_like(rewards)
    next_return = values[-1]  # Bootstrap from last value
    
    for t in reversed(range(len(rewards))):
        returns[t] = rewards[t] + gamma * next_return * (1 - dones[t])
        next_return = returns[t]
    
    return returns
