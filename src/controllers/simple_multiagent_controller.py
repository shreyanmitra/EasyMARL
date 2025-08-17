"""
(C) Shreyan Mitra, based on starter code by Natasha Jaques

Simple Multi-Agent Controller - Designed for MARL Beginners

This controller provides a beginner-friendly interface to multi-agent reinforcement
learning. It maintains the simplicity and clarity of educational code while providing
access to all the advanced features of the EasyMARL framework.

Educational Design Principles:
1. 📚 Simple, descriptive method names that explain their purpose
2. 📝 Extensive comments explaining every step and concept
3. 🔄 Clear, logical flow that matches how beginners think about RL
4. 🎯 Focused on learning rather than optimization
5. 🧩 Modular structure that's easy to understand and modify

Key Features for Beginners:
✅ Step-by-step episode execution with clear phases
✅ Detailed logging to understand what's happening
✅ Simple parameter tuning interface
✅ Built-in visualization and analysis tools
✅ Comprehensive error handling with helpful messages

When to Use This Controller:
👨‍🎓 Learning MARL for the first time
👩‍🏫 Teaching MARL concepts
🔬 Prototyping and experimentation
📊 Research that needs clear, auditable code
🐛 Debugging algorithm behavior

Comparison with ModernMultiAgentController:
- Simple: Optimized for understanding and learning
- Modern: Optimized for performance and scalability
- Both: Support all the same algorithms and features

For MARL Beginners:
Start here! This controller guides you through each step of multi-agent learning
with clear explanations. Once you understand the concepts, you can graduate to
the ModernMultiAgentController for production use.

Workflow Overview:
1. Initialize → Create agents and environment
2. Train → Run episodes with clear learning phases
3. Evaluate → Test performance with visualizations
4. Analyze → Understand what the agents learned
"""

# =============================================================================
# IMPORTS: Required libraries and framework components
# =============================================================================

import torch                # PyTorch for deep learning and neural networks
import numpy as np          # Numerical computations and array operations
import wandb                # Weights & Biases for experiment tracking and visualization
import os                   # Operating system interface for file operations
from typing import Dict, Any, Optional, List  # Type hints for better code clarity and IDE support

# EasyMARL Framework Components
from algorithms import create_marl_algorithm, list_available_algorithms  # Algorithm factory and registry
from utils import plot_single_frame, make_video  # Visualization and video creation utilities


# =============================================================================
# MAIN CONTROLLER CLASS: Simple structure optimized for learning
# =============================================================================

class SimpleMultiAgentController:
    """
    A Simple, Educational Multi-Agent Reinforcement Learning Controller.
    
    This controller is specifically designed for MARL beginners and educational use.
    It provides a clear, step-by-step approach to multi-agent learning while
    maintaining access to all the advanced features of the EasyMARL framework.
    
    Educational Structure:
    1. 🚀 Initialization: Set up agents, environment, and learning system
    2. 📚 Training Phase: Learn through repeated environment interaction
    3. 🎯 Evaluation Phase: Test what the agents learned
    4. 📊 Analysis Phase: Visualize and understand the results
    
    Key Methods (in order of typical use):
    - __init__(): Set up the entire learning system
    - train(): Run the complete training process
    - run_one_episode(): Execute a single learning episode (called by train)
    - evaluate(): Test the trained agents' performance
    - visualize(): Create videos and plots of agent behavior
    - save_models() / load_models(): Persist learned knowledge
    
    For Beginners:
    Think of this as your "MARL tutorial" - it walks you through each step
    of the learning process with clear explanations of what's happening and why.
    """
    """
    
    def __init__(self, env, config: Dict, device: torch.device, 
                 algorithm: str = 'ippo', training: bool = True, 
                 debug: bool = False):
        """
        Initialize the multi-agent controller.
        
        This is similar to the original MultiAgent.__init__ but with
        support for multiple algorithms and enhanced features.
        
        Args:
            env: Multi-agent environment (like MultiGrid)
            config: Configuration dictionary with hyperparameters
            device: PyTorch device (CPU or GPU)
            algorithm: Name of MARL algorithm to use (e.g., 'qmix', 'ippo')
            training: Whether this is for training or evaluation
            debug: Whether to enable debug mode with extra logging
        """
        # Store basic configuration - similar to original metacontroller
        self.env = env                    # Environment for training
        self.config = config              # Configuration dictionary
        self.device = device              # Computing device (CPU/GPU)
        self.algorithm_name = algorithm.lower()  # Name of selected algorithm
        self.training = training          # Training vs evaluation mode
        self.debug = debug               # Debug mode flag
        
        # Validate environment - ensure it has required attributes
        if not hasattr(env, 'n_agents'):
            raise ValueError("Environment must have 'n_agents' attribute")
        
        self.n_agents = env.n_agents     # Number of agents in environment
        
        # Create the MARL algorithm - this replaces the original getAgentClass()
        try:
            self.algorithm = create_marl_algorithm(
                self.algorithm_name, env, config, device
            )
        except ValueError as e:
            print(f"Error creating algorithm: {e}")
            print("Available algorithms:")
            list_available_algorithms()
            raise
        
        # Training statistics - similar to original but more comprehensive
        self.episode_count = 0           # Number of episodes completed
        self.total_steps = 0             # Total environment steps taken
        self.best_performance = float('-inf')  # Best reward achieved
        
        # Logging and tracking - enhanced from original
        self.episode_rewards = []        # List of episode rewards
        self.episode_lengths = []        # List of episode lengths
        self.current_log = ""           # Current episode log (like original)
        
        # Print initialization info - helpful for beginners
        print(f"Initialized SimpleMultiAgentController")
        print(f"Algorithm: {self.algorithm_name.upper()}")
        print(f"Number of agents: {self.n_agents}")
        print(f"Training mode: {self.training}")
        print(f"Device: {device}")
    
    def run_one_episode(self, episode: int, log: bool = True, 
                       train: bool = True, save_model: bool = True, 
                       visualize: bool = False) -> Dict[str, Any]:
        """
        Run a single episode of training or evaluation.
        
        This method is structured very similarly to the original
        multiagent_metacontroller.run_one_episode() but with
        enhanced functionality and better error handling.
        
        Args:
            episode: Current episode number
            log: Whether to log episode details
            train: Whether to train the agents after the episode
            save_model: Whether to save models periodically
            visualize: Whether to collect visualization data
            
        Returns:
            Dictionary containing episode results and metrics
        """
        # Initialize episode tracking variables
        episode_length = 0               # Count of steps in this episode
        rewards = []                     # List of rewards for each step
        episode_reward = 0              # Total reward for this episode
        agent_rewards = [0] * self.n_agents  # Individual agent rewards
        
        # Reset environment to start new episode
        state = self.env.reset()        # Get initial state from environment
        done = False                    # Episode completion flag
        
        # Initialize visualization data if requested
        viz_data = {}
        if visualize:
            viz_data['actions'] = []                # Actions taken by agents
            viz_data['agents_partial_images'] = [] # Agent partial observations
            viz_data['full_images'] = []           # Full environment images
            viz_data['predicted_actions'] = []     # Action predictions (if available)
        
        # Clear current episode log
        self.current_log = ""
        
        # Main episode loop - run until episode ends or max steps reached
        while not done and episode_length < self.config.get('max_steps', 100):
            # Render environment if logging is enabled
            if log:
                self.env.render()
            
            # Increment step counter
            episode_length += 1
            
            # Get actions from all agents - this replaces the original agent loop
            actions = self._get_actions_from_all_agents(state, training=train)
            
            # Take environment step with all agent actions
            next_state, reward, done, info = self.env.step(actions)
            
            # Process and store rewards
            if isinstance(reward, list):
                # Multiple agents with individual rewards
                step_reward = sum(reward)
                for i, r in enumerate(reward):
                    agent_rewards[i] += r
            else:
                # Single reward for all agents
                step_reward = reward
                for i in range(self.n_agents):
                    agent_rewards[i] += reward
            
            rewards.append(reward)       # Store step rewards
            episode_reward += step_reward  # Add to total episode reward
            
            # Log current step - similar to original currentLog
            self.current_log += (
                f"Episode clock: {episode_length} "
                f"Actions: {actions} "
                f"Rewards: {reward} "
                f"Done? {done}\n\n"
            )
            
            # Store visualization data if requested
            if visualize:
                viz_data = self._add_visualization_data(
                    viz_data, self.env, state, actions, next_state
                )
            
            # Store transitions in agent memories
            self._store_transitions_in_agents(state, actions, reward, next_state, done)
            
            # Update state for next iteration
            state = next_state
        
        # Store episode statistics in agent memories (like original)
        for i in range(self.n_agents):
            current_agent = self.algorithm.agents[i]
            
            # Add episode length to agent memory
            if "eps_length" not in current_agent.memory.keys():
                current_agent.memory["eps_length"] = []
            current_agent.memory["eps_length"].append(episode_length)
            
            # Add rewards to agent memory
            if "rewards" not in current_agent.memory.keys():
                current_agent.memory["rewards"] = []
            current_agent.memory["rewards"].append(agent_rewards[i])
        
        # Logging and checkpointing - similar to original
        if log:
            self.log_one_episode(episode, episode_length, rewards)
        
        self.print_terminal_output(episode, episode_reward)
        
        # Save model checkpoints if requested
        if save_model:
            self.save_model_checkpoints(episode)
        
        # Update statistics
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        self.total_steps += episode_length
        
        # Track best performance
        if episode_reward > self.best_performance:
            self.best_performance = episode_reward
        
        # Clear current log
        self.current_log = ""
        
        # Return visualization data if requested
        if visualize:
            viz_data['rewards'] = np.array(rewards)
            return viz_data
        
        # Train agents if requested
        if train:
            self.update_models()
        
        # Return episode statistics
        return {
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'agent_rewards': agent_rewards,
            'total_steps': self.total_steps
        }
    
    def _get_actions_from_all_agents(self, state, training: bool = True) -> List[int]:
        """
        Get actions from all agents given the current state.
        
        This replaces the original loop that called agent.get_action_predictions()
        but works with our modern algorithm framework.
        
        Args:
            state: Current environment state
            training: Whether agents are in training mode
            
        Returns:
            List of actions for all agents
        """
        actions = []
        
        # Get action from each agent
        for i in range(self.n_agents):
            # Extract individual agent observation from global state
            agent_obs = self._extract_agent_observation(state, i)
            
            # Get action from agent using algorithm-specific method
            if hasattr(self.algorithm.agents[i], 'get_action'):
                try:
                    # Different algorithms have different get_action signatures
                    if self.algorithm_name in ['mappo']:
                        # MAPPO needs global state
                        global_state = self._get_global_state(state)
                        action, _, _ = self.algorithm.agents[i].get_action(
                            agent_obs, global_state, training
                        )
                    elif self.algorithm_name in ['maddpg']:
                        # MADDPG returns action probabilities
                        action_probs = self.algorithm.agents[i].get_action(agent_obs, training)
                        action = np.argmax(action_probs)
                    elif self.algorithm_name in ['qmix', 'vdn', 'qtran']:
                        # Value-based methods return action directly
                        action = self.algorithm.agents[i].get_action(agent_obs, training)
                    else:
                        # Default: IPPO and similar algorithms
                        action, _, _ = self.algorithm.agents[i].get_action(agent_obs, training)
                    
                    actions.append(action)
                    
                except Exception as e:
                    # Fallback: random action if agent fails
                    print(f"Agent {i} action selection failed: {e}. Using random action.")
                    action = np.random.randint(0, len(self.env.actions))
                    actions.append(action)
            else:
                # Fallback: random action if no get_action method
                action = np.random.randint(0, len(self.env.actions))
                actions.append(action)
        
        return actions
    
    def _extract_agent_observation(self, state, agent_id: int):
        """
        Extract individual agent observation from global state.
        
        Args:
            state: Global environment state
            agent_id: ID of the agent
            
        Returns:
            Individual agent observation
        """
        if isinstance(state, dict):
            agent_obs = {}
            for key, value in state.items():
                if isinstance(value, list):
                    # Multi-agent observation: select agent's observation
                    agent_obs[key] = value[agent_id] if agent_id < len(value) else value[0]
                else:
                    # Single observation: use for all agents
                    agent_obs[key] = value
        else:
            # List or single observation
            agent_obs = state[agent_id] if isinstance(state, list) else state
        
        return agent_obs
    
    def _get_global_state(self, state) -> np.ndarray:
        """
        Create global state representation for algorithms that need it.
        
        Args:
            state: Current environment state
            
        Returns:
            Global state as numpy array
        """
        if isinstance(state, dict):
            state_parts = []
            for i in range(self.n_agents):
                agent_obs = self._extract_agent_observation(state, i)
                # Flatten observation
                obs_flat = []
                for key, value in agent_obs.items():
                    if isinstance(value, np.ndarray):
                        obs_flat.append(value.flatten())
                    else:
                        obs_flat.append(np.array([value]))
                state_parts.append(np.concatenate(obs_flat))
            return np.concatenate(state_parts)
        else:
            return np.array(state).flatten()
    
    def _store_transitions_in_agents(self, state, actions, rewards, next_state, done):
        """
        Store transitions in agent memories for training.
        
        Args:
            state: Current state
            actions: Actions taken
            rewards: Rewards received
            next_state: Next state
            done: Episode done flag
        """
        for i, agent in enumerate(self.algorithm.agents):
            if hasattr(agent, 'store_transition'):
                agent_obs = self._extract_agent_observation(state, i)
                agent_reward = rewards[i] if isinstance(rewards, list) else rewards
                
                # Get additional info for storage if available
                log_prob = 0.0
                value = 0.0
                try:
                    _, log_prob, value = agent.get_action(agent_obs, training=False)
                except:
                    pass  # Use default values if get_action fails
                
                # Store transition in agent memory
                agent.store_transition(
                    agent_obs, actions[i], log_prob, value, agent_reward, done
                )
    
    def _add_visualization_data(self, viz_data, env, state, actions, next_state):
        """
        Add visualization data for creating videos.
        
        This is similar to the original add_visualization_data method.
        
        Args:
            viz_data: Existing visualization data
            env: Environment instance
            state: Current state
            actions: Actions taken
            next_state: Next state
            
        Returns:
            Updated visualization data
        """
        viz_data['actions'].append(actions)
        
        # Add agent partial observations if available
        if hasattr(env, 'get_obs_render'):
            partial_images = []
            for i in range(self.n_agents):
                agent_obs = self._extract_agent_observation(state, i)
                if 'image' in agent_obs:
                    partial_images.append(env.get_obs_render(agent_obs['image']))
            viz_data['agents_partial_images'].append(partial_images)
        
        # Add full environment image
        if hasattr(env, 'render'):
            viz_data['full_images'].append(env.render('rgb_array'))
        
        # Add predicted actions if model supports it
        if hasattr(self.algorithm, 'get_action_predictions'):
            viz_data['predicted_actions'].append(
                self.algorithm.get_action_predictions(next_state)
            )
        
        return viz_data
    
    def log_one_episode(self, episode: int, episode_length: int, rewards: List):
        """
        Log details of one episode.
        
        This is similar to the original log_one_episode method.
        
        Args:
            episode: Episode number
            episode_length: Number of steps in episode
            rewards: List of rewards received
        """
        print(self.current_log)  # Print detailed episode log
        
        # Log to WandB if available
        if wandb.run is not None:
            total_reward = sum(rewards) if isinstance(rewards[0], (int, float)) else sum(sum(r) for r in rewards)
            wandb.log({
                'episode/reward': total_reward,
                'episode/length': episode_length,
                'episode/number': episode
            })
    
    def print_terminal_output(self, episode: int, total_reward: float):
        """
        Print training progress to terminal.
        
        This is identical to the original print_terminal_output method.
        
        Args:
            episode: Episode number
            total_reward: Total reward for episode
        """
        if episode % self.config.get('print_every', 100) == 0:
            print('Total steps: {} \t Episode: {} \t Total reward: {}'.format(
                self.total_steps, episode, total_reward))
    
    def save_model_checkpoints(self, episode: int):
        """
        Save model checkpoints periodically.
        
        This is similar to the original save_model_checkpoints method.
        
        Args:
            episode: Current episode number
        """
        if episode % self.config.get('save_model_episode', 1000) == 0 and episode > 0:
            # Save using algorithm's method
            if hasattr(self.algorithm, 'save_models'):
                self.algorithm.save_models(f"episode_{episode}")
            else:
                # Fallback: save individual agents
                for i, agent in enumerate(self.algorithm.agents):
                    if hasattr(agent, 'save_model'):
                        agent.save_model(f"checkpoint_episode_{episode}_agent_{i}")
    
    def update_models(self):
        """
        Update agent models after collecting enough experience.
        
        This replaces the original update_models method but with
        support for different algorithm types.
        """
        # Only update if we have enough experience
        if self.total_steps > self.config.get('initial_memory', 1000):
            if self.total_steps % self.config.get('update_every', 100) == 0:
                # Update using algorithm's training method
                if hasattr(self.algorithm, 'train_step'):
                    # Create rollout data for modern algorithms
                    rollout_data = {
                        'total_steps': self.total_steps,
                        'episode_count': self.episode_count
                    }
                    metrics = self.algorithm.train_step(rollout_data)
                else:
                    # Fallback: update individual agents
                    for i, agent in enumerate(self.algorithm.agents):
                        if hasattr(agent, 'update'):
                            agent.update()
    
    def train(self, total_episodes: int = None):
        """
        Run the complete training loop.
        
        This is structured similarly to the original train method
        but with enhanced features and better error handling.
        
        Args:
            total_episodes: Number of episodes to train (uses config if None)
        """
        if not self.training:
            raise NotImplementedError("Cannot train an agent configured with train = False")
        
        # Get number of episodes from config if not specified
        n_episodes = total_episodes or self.config.get('n_episodes', 1000)
        
        print(f"Starting training for {n_episodes} episodes...")
        print(f"Using {self.algorithm_name.upper()} algorithm")
        
        # Initialize WandB if configured
        if self.config.get('use_wandb', False):
            wandb.init(
                project=self.config.get('wandb_project', 'easymarl'),
                name=f"{self.algorithm_name}_{self.config.get('environment', 'unknown')}",
                config=self.config
            )
        
        # Main training loop
        for episode in range(n_episodes):
            # Run visualization episodes periodically
            if episode % self.config.get('visualize_every', 1000) == 0 and not (self.debug and episode == 0):
                viz_data = self.run_one_episode(episode, visualize=True)
                self.visualize(self.env, f'{self.algorithm_name}_training_step_{episode}', viz_data=viz_data)
            else:
                # Regular training episode
                self.run_one_episode(episode)
            
            # Update episode counter
            self.episode_count += 1
        
        # Training completed
        print(f"Training completed after {n_episodes} episodes")
        print(f"Best performance: {self.best_performance:.2f}")
        
        # Close environment and WandB
        self.env.close()
        if wandb.run is not None:
            wandb.finish()
    
    def visualize(self, env, mode: str, video_dir: str = 'videos', viz_data: Dict = None):
        """
        Create visualization video of agent behavior.
        
        This is identical to the original visualize method.
        
        Args:
            env: Environment instance
            mode: Name for the video file
            video_dir: Directory to save videos
            viz_data: Visualization data (collected if None)
        """
        if not viz_data:
            viz_data = self.run_one_episode(
                episode=0, log=False, train=False, save_model=False, visualize=True
            )
            env.close()
        
        # Set up video directory
        video_path = os.path.join(video_dir, self.config.get('experiment_name', 'experiment'), 
                                 self.config.get('model_name', self.algorithm_name))
        
        if not os.path.exists(video_path):
            os.makedirs(video_path)
        
        # Get action names for visualization
        action_dict = {}
        if hasattr(env, 'Actions'):
            for act in env.Actions:
                action_dict[act.value] = act.name
        else:
            # Default action names
            for i in range(len(env.actions)):
                action_dict[i] = f"Action_{i}"
        
        # Create video frames
        traj_len = len(viz_data['rewards'])
        for t in range(traj_len):
            self.visualize_one_frame(t, viz_data, action_dict, video_path, self.algorithm_name)
            print(f'Frame {t}/{traj_len}')
        
        # Create final video
        make_video(video_path, f'{mode}_trajectory_video')
    
    def visualize_one_frame(self, t: int, viz_data: Dict, action_dict: Dict, 
                           video_path: str, model_name: str):
        """
        Create one frame of visualization.
        
        This is identical to the original visualize_one_frame method.
        
        Args:
            t: Time step
            viz_data: Visualization data
            action_dict: Mapping of action IDs to names
            video_path: Path to save video frames
            model_name: Name of the model
        """
        plot_single_frame(
            t,
            viz_data['full_images'][t],
            viz_data['agents_partial_images'][t],
            viz_data['actions'][t],
            viz_data['rewards'],
            action_dict,
            video_path,
            model_name,
            predicted_actions=viz_data.get('predicted_actions'),
            all_actions=viz_data['actions']
        )
    
    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate trained agents.
        
        Args:
            num_episodes: Number of episodes to evaluate
            
        Returns:
            Dictionary of evaluation metrics
        """
        print(f"Evaluating for {num_episodes} episodes...")
        
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(num_episodes):
            episode_data = self.run_one_episode(
                episode, log=False, train=False, save_model=False
            )
            episode_rewards.append(episode_data['episode_reward'])
            episode_lengths.append(episode_data['episode_length'])
        
        # Calculate evaluation metrics
        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'mean_length': np.mean(episode_lengths)
        }
    
    def save_models(self, suffix: str = ""):
        """
        Save all agent models.
        
        Args:
            suffix: Suffix for save file names
        """
        save_path = os.path.join(
            self.config.get('model_save_path', 'models'),
            f"{self.algorithm_name}_{suffix}"
        )
        
        if hasattr(self.algorithm, 'save_models'):
            self.algorithm.save_models(save_path)
        else:
            for i, agent in enumerate(self.algorithm.agents):
                if hasattr(agent, 'save_model'):
                    agent.save_model(f"{save_path}_agent_{i}")
        
        print(f"Models saved to {save_path}")
    
    def load_models(self, suffix: str = ""):
        """
        Load all agent models.
        
        Args:
            suffix: Suffix of save file names
        """
        load_path = os.path.join(
            self.config.get('model_save_path', 'models'),
            f"{self.algorithm_name}_{suffix}"
        )
        
        if hasattr(self.algorithm, 'load_models'):
            self.algorithm.load_models(load_path)
        else:
            for i, agent in enumerate(self.algorithm.agents):
                if hasattr(agent, 'load_model'):
                    agent.load_model(f"{load_path}_agent_{i}")
        
        print(f"Models loaded from {load_path}")


# =============================================================================
# COMPATIBILITY ALIAS: For backward compatibility
# =============================================================================

# Create an alias so existing code can still use "MultiAgent" class name
MultiAgent = SimpleMultiAgentController
