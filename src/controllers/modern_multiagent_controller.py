"""
(C) Shreyan Mitra, based on starter code by Natasha Jaques

Modern Multi-Agent Controller for EasyMARL Framework

This is the main orchestrator that manages multi-agent training and evaluation.
It provides a clean, modular interface for working with different MARL algorithms
and handles all the complex coordination between agents, environments, and logging.

Key Responsibilities:
1. Algorithm Management: Creates and manages MARL algorithm instances
2. Training Coordination: Orchestrates the training loop across multiple agents
3. Environment Interaction: Handles agent-environment interactions
4. Data Collection: Manages experience collection and replay buffers
5. Performance Tracking: Logs metrics and creates visualizations
6. Model Persistence: Saves and loads trained models

Features:
✅ Supports 20+ MARL algorithms (IPPO, QMIX, MADDPG, MAPPO, etc.)
✅ Automatic experiment tracking with Weights & Biases
✅ Built-in evaluation and visualization
✅ Modular design for easy algorithm swapping
✅ Robust error handling and debugging support

For MARL Beginners:
Think of this as the "conductor" of an orchestra. While individual agents (musicians)
have their own skills, the controller coordinates everything to create a harmonious
performance. It handles all the complex logistics so you can focus on the algorithms.

Architecture:
- Controller ← Creates/Manages → Algorithm ← Contains → Agents
- Controller ← Interacts → Environment
- Controller ← Logs → Metrics & Visualizations
"""

# Import necessary libraries
import torch          # PyTorch for deep learning
import numpy as np    # Numerical computations
import wandb          # Weights & Biases for experiment tracking
import os             # Operating system interface
from typing import Dict, Any, Optional  # Type hints for better code clarity
from PIL import Image # Python Imaging Library for image processing

# Import framework components
from algorithms import create_marl_algorithm, list_available_algorithms  # Algorithm factory and registry
from utils import plot_single_frame, make_video  # Visualization utilities
from research_interface import get_research_interface, ExperimentConfig  # Research features


class ModernMultiAgentController:
    """
    Modern Multi-Agent Reinforcement Learning Controller.
    
    This class serves as the central coordinator for multi-agent training and evaluation.
    It abstracts away the complexity of managing multiple agents, different algorithms,
    and various training procedures behind a clean, easy-to-use interface.
    
    Key Design Principles:
    1. Algorithm Agnostic: Works with any MARL algorithm that follows our interface
    2. Environment Agnostic: Works with any multi-agent environment
    3. Modular: Easy to extend and customize
    4. Robust: Handles errors gracefully and provides helpful debugging
    
    Workflow:
    1. Initialize with environment and configuration
    2. Create specified MARL algorithm and agents
    3. Run training loop with automatic logging
    4. Evaluate performance and create visualizations
    5. Save/load models for persistence
    
    For MARL Beginners:
    This is your main interface to the framework. You specify what algorithm
    you want to use and what environment to train on, and this controller
    handles all the complex details.
    """
    
    def __init__(self, env, config: Dict, device: torch.device, 
                 algorithm: str = 'ippo', training: bool = True, debug: bool = False):
        """
        Initialize the multi-agent controller with specified configuration.
        
        This sets up the entire multi-agent learning system: creates the algorithm,
        initializes agents, sets up logging, and prepares for training or evaluation.
        
        Args:
            env: Multi-agent environment that follows Gym interface
                 Must have attributes: n_agents, observation_space, action_space
            config (Dict): Configuration dictionary containing hyperparameters
                          Example: {'gamma': 0.99, 'lr': 3e-4, 'episodes': 10000, ...}
            device (torch.device): Computing device (CPU or GPU)
                                  Example: torch.device('cuda') or torch.device('cpu')
            algorithm (str): Name of MARL algorithm to use
                           Options: 'ippo', 'qmix', 'maddpg', 'mappo', etc.
            training (bool): Whether this is for training (True) or evaluation (False)
            debug (bool): Enable debug mode for additional logging and error checking
        
        For Beginners:
        This is like setting up a classroom: you specify the teaching method (algorithm),
        the students (agents), the subject (environment), and the learning materials (config).
        """
        # Store core components
        self.env = env                    # Multi-agent environment
        self.config = config              # Learning configuration
        self.device = device              # Computing device (CPU/GPU)
        self.algorithm_name = algorithm.lower()  # Normalize algorithm name
        self.training = training          # Training vs evaluation mode
        self.debug = debug               # Debug mode flag
        
        # Validate environment compatibility
        if not hasattr(env, 'n_agents'):
            raise ValueError("Environment must have 'n_agents' attribute for multi-agent support")
        
        # Validate algorithm availability
        available_algorithms = list_available_algorithms()
        if self.algorithm_name not in available_algorithms:
            raise ValueError(f"Algorithm '{algorithm}' not available. "
                           f"Available algorithms: {available_algorithms}")
        
        print(f"Initializing ModernMultiAgentController")
        print(f"Algorithm: {self.algorithm_name}")
        print(f"Environment: {env.__class__.__name__} with {env.n_agents} agents")
        print(f"Device: {device}")
        print(f"Mode: {'Training' if training else 'Evaluation'}")
        
        # Initialize training metrics
        self.episode_count = 0           # Number of episodes completed
        self.total_steps = 0             # Total environment steps taken
        self.best_performance = -float('inf')  # Best performance achieved
        
        # Initialize storage for visualization and analysis
        self.episode_rewards = []        # Reward history for plotting
        self.episode_lengths = []        # Episode length history
        self.training_metrics = []       # Detailed training metrics
        
        self.n_agents = env.n_agents
        
        # Create the MARL algorithm
        try:
            self.algorithm = create_marl_algorithm(
                self.algorithm_name, env, config, device
            )
        except ValueError as e:
            print(f"Error creating algorithm: {e}")
            print("Available algorithms:")
            list_available_algorithms()
            raise
        
        # Training statistics
        self.episode_count = 0
        self.total_steps = 0
        self.best_performance = float('-inf')
        
        # Logging
        self.episode_rewards = []
        self.episode_lengths = []
        
        print(f"Initialized ModernMultiAgentController")
        print(f"Algorithm: {self.algorithm_name.upper()}")
        print(f"Number of agents: {self.n_agents}")
        print(f"Training mode: {self.training}")
        print(f"Device: {device}")
    
    def train(self, total_episodes: int) -> None:
        """
        Train the multi-agent system.
        
        Args:
            total_episodes: Total number of episodes to train for
        """
        if not self.training:
            raise ValueError("Controller not initialized for training")
        
        print(f"Starting training for {total_episodes} episodes...")
        print(f"Using {self.algorithm_name.upper()} algorithm")
        
        # Initialize WandB if configured
        if self.config.get('use_wandb', False):
            wandb.init(
                project=self.config.get('wandb_project', 'easymarl'),
                name=f"{self.algorithm_name}_{self.config.get('environment', 'unknown')}",
                config=self.config
            )
        
        for episode in range(total_episodes):
            # Run one episode and get detailed metrics
            episode_data = self.run_one_episode(episode)
            
            # Update statistics
            self.episode_count += 1
            episode_reward = episode_data.get('total_reward', 0)
            episode_length = episode_data.get('episode_length', 0)
            
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.total_steps += episode_length
            
            # Track best performance
            if episode_reward > self.best_performance:
                self.best_performance = episode_reward
            
            # Logging
            if episode % self.config.get('log_interval', 100) == 0:
                metrics = {
                    'episode_reward': episode_reward,
                    'episode_length': episode_length,
                    'total_steps': self.total_steps,
                    **episode_data.get('training_metrics', {})
                }
                self._log_training_progress(episode, metrics)
            
            # Save models
            if episode % self.config.get('save_interval', 1000) == 0 and episode > 0:
                self.save_models(f"episode_{episode}")
            
            # Generate visualization
            if episode % self.config.get('visualize_every', 1000) == 0 and not self.debug:
                self.visualize_episode(episode)
            
            # Evaluation
            if episode % self.config.get('eval_interval', 500) == 0 and episode > 0:
                eval_metrics = self.evaluate(num_episodes=10)
                self._log_evaluation(episode, eval_metrics)
        
        print(f"Training completed after {total_episodes} episodes")
        print(f"Best performance: {self.best_performance:.2f}")
        
        # Final model save
        self.save_models("final")
        
        # Close WandB
        if wandb.run is not None:
            wandb.finish()
    
    def run_one_episode(self, episode: int) -> Dict[str, Any]:
        """
        Run a single episode and return comprehensive metrics.
        
        Args:
            episode: Current episode number
            
        Returns:
            Dictionary containing episode metrics and data
        """
        obs = self.env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        episode_rewards = []
        episode_actions = []
        episode_observations = []
        agent_rewards = [0] * self.n_agents
        
        # Episode tracking variables
        current_log = ""
        
        while not done and episode_length < self.config.get('max_steps', 100):
            # Store current observation
            episode_observations.append(obs)
            
            # Get actions from all agents
            actions = self._get_actions(obs, training=True)
            episode_actions.append(actions)
            
            # Log current step
            current_log += f"Step {episode_length}: Actions: {actions}"
            
            # Take environment step
            next_obs, rewards, done, info = self.env.step(actions)
            
            # Process rewards
            if isinstance(rewards, list):
                step_reward = sum(rewards)
                episode_rewards.append(rewards)
                for i, r in enumerate(rewards):
                    agent_rewards[i] += r
            else:
                step_reward = rewards
                episode_rewards.append([rewards] * self.n_agents)
                for i in range(self.n_agents):
                    agent_rewards[i] += rewards
            
            current_log += f" Rewards: {rewards} Done: {done}\n"
            
            episode_reward += step_reward
            episode_length += 1
            
            # Store transitions in agent memories (if supported)
            for i, agent in enumerate(self.algorithm.agents):
                if hasattr(agent, 'store_transition'):
                    agent_obs = self._extract_agent_obs(obs, i)
                    agent_next_obs = self._extract_agent_obs(next_obs, i)
                    agent_reward = rewards[i] if isinstance(rewards, list) else rewards
                    
                    # Get action log prob and value if available
                    log_prob = 0.0
                    value = 0.0
                    if hasattr(agent, 'get_action'):
                        try:
                            _, log_prob, value = agent.get_action(agent_obs, training=False)
                        except:
                            pass
                    
                    agent.store_transition(
                        agent_obs, actions[i], log_prob, value, agent_reward, done
                    )
            
            obs = next_obs
        
        # Training step (update models)
        training_metrics = {}
        if hasattr(self.algorithm, 'train_step'):
            # Create rollout data
            rollout_data = {
                'observations': episode_observations,
                'actions': episode_actions,
                'rewards': episode_rewards,
                'episode_length': episode_length,
                'total_reward': episode_reward,
                'agent_rewards': agent_rewards
            }
            training_metrics = self.algorithm.train_step(rollout_data)
        
        # Individual agent updates (if supported)
        agent_metrics = {}
        for i, agent in enumerate(self.algorithm.agents):
            if hasattr(agent, 'update') and hasattr(agent, 'memory'):
                if len(getattr(agent, 'memory', {}).get('rewards', [])) > 0:
                    try:
                        agent_update_metrics = agent.update()
                        for key, value in agent_update_metrics.items():
                            if key not in agent_metrics:
                                agent_metrics[key] = []
                            agent_metrics[key].append(value)
                    except Exception as e:
                        print(f"Agent {i} update failed: {e}")
        
        # Average agent metrics
        averaged_agent_metrics = {}
        for key, values in agent_metrics.items():
            if values:
                averaged_agent_metrics[f'agent_{key}_mean'] = np.mean(values)
                averaged_agent_metrics[f'agent_{key}_std'] = np.std(values)
        
        # Combine all metrics
        all_metrics = {**training_metrics, **averaged_agent_metrics}
        
        # Log episode details (similar to original metacontroller)
        if episode % self.config.get('print_every', 100) == 0:
            print(f"Episode {episode}: Total Reward: {episode_reward:.2f}, Length: {episode_length}")
            if self.debug:
                print(current_log)
        
        # WandB logging
        if wandb.run is not None:
            wandb_metrics = {
                'episode/reward': episode_reward,
                'episode/length': episode_length,
                'episode/x_axis': episode,
                'step/total_steps': self.total_steps + episode_length,
                **{f'episode/{k}': v for k, v in all_metrics.items() if isinstance(v, (int, float))}
            }
            
            # Log individual agent rewards
            for i, reward in enumerate(agent_rewards):
                wandb_metrics[f'agent_{i}/reward'] = reward
            
            wandb.log(wandb_metrics)
        
        return {
            'total_reward': episode_reward,
            'episode_length': episode_length,
            'agent_rewards': agent_rewards,
            'training_metrics': all_metrics,
            'episode_log': current_log,
            'observations': episode_observations,
            'actions': episode_actions,
            'rewards': episode_rewards
        }
    
    def evaluate(self, num_episodes: int = 10, render: bool = False) -> Dict[str, float]:
        """
        Evaluate the current policy.
        
        Args:
            num_episodes: Number of episodes to evaluate
            render: Whether to render during evaluation
            
        Returns:
            Dictionary containing evaluation metrics
        """
        print(f"Evaluating policy for {num_episodes} episodes...")
        
        if hasattr(self.algorithm, 'evaluate'):
            # Use algorithm's built-in evaluation
            return self.algorithm.evaluate(self.env, num_episodes)
        else:
            # Generic evaluation
            return self._generic_evaluation(num_episodes, render)
    
    def _generic_evaluation(self, num_episodes: int, render: bool = False) -> Dict[str, float]:
        """Generic evaluation implementation."""
        episode_rewards = []
        episode_lengths = []
        success_rate = 0
        
        for episode in range(num_episodes):
            obs = self.env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            
            while not done:
                if render:
                    self.env.render()
                
                # Get actions (evaluation mode)
                actions = self._get_actions(obs, training=False)
                
                # Take step
                obs, rewards, done, info = self.env.step(actions)
                
                # Update metrics
                if isinstance(rewards, list):
                    episode_reward += sum(rewards)
                else:
                    episode_reward += rewards
                
                episode_length += 1
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # Check for success (if info provides it)
            if isinstance(info, dict) and info.get('success', False):
                success_rate += 1
        
        # Compute statistics
        eval_metrics = {
            'eval_mean_reward': np.mean(episode_rewards),
            'eval_std_reward': np.std(episode_rewards),
            'eval_min_reward': np.min(episode_rewards),
            'eval_max_reward': np.max(episode_rewards),
            'eval_mean_length': np.mean(episode_lengths),
            'eval_success_rate': success_rate / num_episodes
        }
        
        return eval_metrics
    
    def _get_actions(self, obs, training: bool = True) -> list:
        """Get actions from all agents."""
        actions = []
        
        # Extract individual observations for each agent
        for i in range(self.n_agents):
            agent_obs = self._extract_agent_obs(obs, i)
            
            # Get action from agent
            if hasattr(self.algorithm.agents[i], 'get_action'):
                if self.algorithm_name in ['mappo']:
                    # MAPPO needs global state
                    global_state = self._get_global_state(obs)
                    action, _, _ = self.algorithm.agents[i].get_action(
                        agent_obs, global_state, training
                    )
                else:
                    # Other algorithms
                    if self.algorithm_name in ['maddpg']:
                        action_probs = self.algorithm.agents[i].get_action(agent_obs, training)
                        action = np.argmax(action_probs)
                    elif self.algorithm_name in ['qmix']:
                        action = self.algorithm.agents[i].get_action(agent_obs, training)
                    else:  # IPPO and others
                        action, _, _ = self.algorithm.agents[i].get_action(agent_obs, training)
            else:
                # Fallback: random action
                action = np.random.randint(0, len(self.env.actions))
            
            actions.append(action)
        
        return actions
    
    def _extract_agent_obs(self, obs, agent_id: int):
        """Extract observation for a specific agent."""
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
    
    def _get_global_state(self, obs) -> np.ndarray:
        """Get global state representation."""
        if hasattr(self.algorithm, '_get_global_state'):
            return self.algorithm._get_global_state(obs)
        else:
            # Default implementation
            if isinstance(obs, dict):
                state_parts = []
                for i in range(self.n_agents):
                    agent_obs = self._extract_agent_obs(obs, i)
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
                return np.array(obs).flatten()
    
    def visualize_episode(self, episode_num: int = 0) -> None:
        """
        Generate a visualization of an episode.
        
        Args:
            episode_num: Episode number for naming
        """
        print(f"Generating visualization for episode {episode_num}...")
        
        # Run episode and collect visualization data
        viz_data = self._collect_visualization_data()
        
        if viz_data:
            # Create video
            video_path = os.path.join(
                self.config.get('video_dir', 'videos'),
                self.config.get('experiment_name', 'experiment'),
                f"{self.algorithm_name}_episode_{episode_num}"
            )
            
            self._create_visualization_video(viz_data, video_path)
    
    def _collect_visualization_data(self) -> Optional[Dict]:
        """Collect data for visualization."""
        obs = self.env.reset()
        done = False
        
        viz_data = {
            'full_images': [],
            'actions': [],
            'rewards': [],
            'agents_partial_images': []
        }
        
        # Collect full image
        if hasattr(self.env, 'render'):
            viz_data['full_images'].append(self.env.render('rgb_array'))
        
        episode_rewards = []
        
        while not done:
            # Get actions
            actions = self._get_actions(obs, training=False)
            viz_data['actions'].append(actions)
            
            # Collect partial observations if available
            if hasattr(self.env, 'get_obs_render'):
                partial_images = []
                for i in range(self.n_agents):
                    agent_obs = self._extract_agent_obs(obs, i)
                    if 'image' in agent_obs:
                        partial_images.append(
                            self.env.get_obs_render(agent_obs['image'])
                        )
                viz_data['agents_partial_images'].append(partial_images)
            
            # Take step
            obs, rewards, done, _ = self.env.step(actions)
            
            # Store rewards
            episode_rewards.append(rewards)
            
            # Collect full image
            if hasattr(self.env, 'render'):
                viz_data['full_images'].append(self.env.render('rgb_array'))
        
        viz_data['rewards'] = np.array(episode_rewards)
        
        return viz_data if viz_data['full_images'] else None
    
    def _create_visualization_video(self, viz_data: Dict, video_path: str) -> None:
        """Create visualization video from collected data."""
        # Set up directory
        if not os.path.exists(video_path):
            os.makedirs(video_path)
        
        # Get action names
        action_dict = {}
        if hasattr(self.env, 'Actions'):
            for act in self.env.Actions:
                action_dict[act.value] = act.name
        else:
            # Default action names
            for i in range(len(self.env.actions)):
                action_dict[i] = f"Action_{i}"
        
        # Generate frames
        traj_len = len(viz_data['rewards'])
        for t in range(traj_len):
            self._create_visualization_frame(
                t, viz_data, action_dict, video_path
            )
        
        # Create video
        if hasattr(viz_data, 'full_images') and viz_data['full_images']:
            make_video(video_path, f"{self.algorithm_name}_trajectory")
    
    def _create_visualization_frame(self, t: int, viz_data: Dict, 
                                  action_dict: Dict, video_path: str) -> None:
        """Create a single visualization frame."""
        if hasattr(plot_single_frame, '__call__'):
            plot_single_frame(
                t,
                viz_data['full_images'][t],
                viz_data.get('agents_partial_images', [[]] * self.n_agents)[t],
                viz_data['actions'][t],
                viz_data['rewards'],
                action_dict,
                video_path,
                f"{self.algorithm_name}_model"
            )
    
    def save_models(self, suffix: str = "") -> None:
        """
        Save all agent models.
        
        Args:
            suffix: Suffix to add to save path
        """
        save_path = os.path.join(
            self.config.get('model_save_path', 'models'),
            self.config.get('experiment_name', 'experiment'),
            f"{self.algorithm_name}_{suffix}"
        )
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save using algorithm's method
        if hasattr(self.algorithm, 'save_models'):
            self.algorithm.save_models(save_path)
        else:
            # Fallback: save individual agents
            for i, agent in enumerate(self.algorithm.agents):
                if hasattr(agent, 'save_model'):
                    agent.save_model(f"{save_path}_agent_{i}")
        
        print(f"Models saved to {save_path}")
    
    def load_models(self, suffix: str = "") -> None:
        """
        Load all agent models.
        
        Args:
            suffix: Suffix of the save path
        """
        load_path = os.path.join(
            self.config.get('model_save_path', 'models'),
            self.config.get('experiment_name', 'experiment'),
            f"{self.algorithm_name}_{suffix}"
        )
        
        # Load using algorithm's method
        if hasattr(self.algorithm, 'load_models'):
            self.algorithm.load_models(load_path)
        else:
            # Fallback: load individual agents
            for i, agent in enumerate(self.algorithm.agents):
                if hasattr(agent, 'load_model'):
                    agent.load_model(f"{load_path}_agent_{i}")
        
        print(f"Models loaded from {load_path}")
    
    def _log_training_progress(self, episode: int, metrics: Dict[str, float]) -> None:
        """Log training progress with comprehensive metrics."""
        # Console logging
        episode_reward = metrics.get('episode_reward', 0)
        episode_length = metrics.get('episode_length', 0)
        total_steps = metrics.get('total_steps', self.total_steps)
        
        print(f"Episode {episode:6d} | "
              f"Reward: {episode_reward:8.2f} | "
              f"Length: {episode_length:4d} | "
              f"Steps: {total_steps:8d}")
        
        # Additional metrics if available
        if 'agent_policy_loss_mean' in metrics:
            print(f"            | Policy Loss: {metrics['agent_policy_loss_mean']:8.4f} | "
                  f"Value Loss: {metrics.get('agent_value_loss_mean', 0):8.4f}")
        
        if 'agent_entropy_loss_mean' in metrics:
            print(f"            | Entropy: {metrics['agent_entropy_loss_mean']:8.4f} | "
                  f"LR: {metrics.get('agent_learning_rate_mean', 0):8.6f}")
        
        # Calculate statistics over recent episodes
        if len(self.episode_rewards) >= 100:
            recent_rewards = self.episode_rewards[-100:]
            mean_reward = np.mean(recent_rewards)
            std_reward = np.std(recent_rewards)
            print(f"            | Avg Reward (100ep): {mean_reward:8.2f} ± {std_reward:6.2f}")
        
        # WandB logging (if not done elsewhere)
        if not self.debug and wandb.run is not None:
            wandb_metrics = {
                'train/episode_reward': episode_reward,
                'train/episode_length': episode_length,
                'train/total_steps': total_steps,
                'episode/x_axis': episode,
                'step/x_axis': total_steps
            }
            
            # Add all metrics with proper prefixes
            for key, value in metrics.items():
                if key not in ['episode_reward', 'episode_length', 'total_steps']:
                    if 'loss' in key.lower():
                        wandb_metrics[f'loss/{key}'] = value
                    elif 'learning_rate' in key.lower():
                        wandb_metrics[f'optim/{key}'] = value
                    else:
                        wandb_metrics[f'train/{key}'] = value
            
            # Add performance statistics
            if len(self.episode_rewards) >= 10:
                recent_rewards = self.episode_rewards[-10:]
                wandb_metrics['performance/reward_mean_10ep'] = np.mean(recent_rewards)
                wandb_metrics['performance/reward_std_10ep'] = np.std(recent_rewards)
            
            if len(self.episode_rewards) >= 100:
                recent_rewards = self.episode_rewards[-100:]
                wandb_metrics['performance/reward_mean_100ep'] = np.mean(recent_rewards)
                wandb_metrics['performance/reward_std_100ep'] = np.std(recent_rewards)
            
            wandb.log(wandb_metrics)
    
    def _log_evaluation(self, episode: int, eval_metrics: Dict[str, float]) -> None:
        """Log evaluation results with enhanced formatting."""
        print(f"\n{'='*50}")
        print(f"Evaluation at episode {episode}:")
        print(f"{'='*50}")
        
        # Group metrics by category
        reward_metrics = {}
        length_metrics = {}
        success_metrics = {}
        other_metrics = {}
        
        for key, value in eval_metrics.items():
            if 'reward' in key.lower():
                reward_metrics[key] = value
            elif 'length' in key.lower():
                length_metrics[key] = value
            elif 'success' in key.lower():
                success_metrics[key] = value
            else:
                other_metrics[key] = value
        
        # Print grouped metrics
        if reward_metrics:
            print("Reward Metrics:")
            for key, value in reward_metrics.items():
                print(f"  {key.replace('_', ' ').title()}: {value:.4f}")
        
        if length_metrics:
            print("Episode Length Metrics:")
            for key, value in length_metrics.items():
                print(f"  {key.replace('_', ' ').title()}: {value:.2f}")
        
        if success_metrics:
            print("Success Metrics:")
            for key, value in success_metrics.items():
                print(f"  {key.replace('_', ' ').title()}: {value:.4f}")
        
        if other_metrics:
            print("Other Metrics:")
            for key, value in other_metrics.items():
                print(f"  {key.replace('_', ' ').title()}: {value:.4f}")
        
        print(f"{'='*50}\n")
        
        # WandB logging
        if not self.debug and wandb.run is not None:
            wandb_eval_metrics = {}
            for key, value in eval_metrics.items():
                wandb_eval_metrics[f'eval/{key}'] = value
            
            wandb_eval_metrics['episode/x_axis'] = episode
            wandb.log(wandb_eval_metrics)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get training statistics."""
        if not self.episode_rewards:
            return {}
        
        return {
            'total_episodes': len(self.episode_rewards),
            'total_steps': self.total_steps,
            'mean_reward': np.mean(self.episode_rewards),
            'std_reward': np.std(self.episode_rewards),
            'min_reward': np.min(self.episode_rewards),
            'max_reward': np.max(self.episode_rewards),
            'best_performance': self.best_performance,
            'mean_episode_length': np.mean(self.episode_lengths),
            'algorithm': self.algorithm_name
        }
    
    def get_research_metrics(self) -> Dict[str, Any]:
        """Get detailed research-oriented metrics and analysis."""
        if not self.episode_rewards:
            return {}
        
        stats = self.get_statistics()
        
        # Add advanced research metrics
        rewards = np.array(self.episode_rewards)
        
        # Performance stability metrics
        if len(rewards) >= 100:
            recent_rewards = rewards[-100:]
            early_rewards = rewards[:100] if len(rewards) > 100 else rewards
            
            stats.update({
                'performance_improvement': np.mean(recent_rewards) - np.mean(early_rewards),
                'stability_coefficient': np.std(recent_rewards) / np.mean(recent_rewards) if np.mean(recent_rewards) != 0 else 0,
                'convergence_rate': self._calculate_convergence_rate(rewards),
                'sample_efficiency': self._calculate_sample_efficiency(),
            })
        
        # Learning curve analysis
        if len(rewards) >= 10:
            stats.update({
                'learning_trend': self._calculate_learning_trend(rewards),
                'plateau_detection': self._detect_plateau(rewards),
                'exploration_decay': self._analyze_exploration_decay()
            })
        
        # Algorithm-specific metrics
        if hasattr(self.algorithm, 'get_research_metrics'):
            algo_metrics = self.algorithm.get_research_metrics()
            stats.update({f'algorithm_{k}': v for k, v in algo_metrics.items()})
        
        return stats
    
    def _calculate_convergence_rate(self, rewards: np.ndarray) -> float:
        """Calculate the rate of convergence based on reward improvement."""
        if len(rewards) < 50:
            return 0.0
        
        # Fit a line to recent rewards and calculate slope
        x = np.arange(len(rewards))
        coeffs = np.polyfit(x, rewards, 1)
        return float(coeffs[0])  # Slope indicates convergence rate
    
    def _calculate_sample_efficiency(self) -> float:
        """Calculate sample efficiency metric."""
        if not self.episode_rewards or self.total_steps == 0:
            return 0.0
        
        # Simple metric: reward improvement per step
        if len(self.episode_rewards) >= 2:
            improvement = self.episode_rewards[-1] - self.episode_rewards[0]
            return improvement / self.total_steps
        return 0.0
    
    def _calculate_learning_trend(self, rewards: np.ndarray) -> str:
        """Analyze the current learning trend."""
        if len(rewards) < 20:
            return "insufficient_data"
        
        recent = rewards[-20:]
        trend = np.polyfit(range(len(recent)), recent, 1)[0]
        
        if trend > 0.1:
            return "improving"
        elif trend < -0.1:
            return "degrading"
        else:
            return "stable"
    
    def _detect_plateau(self, rewards: np.ndarray, window_size: int = 50) -> bool:
        """Detect if learning has plateaued."""
        if len(rewards) < window_size * 2:
            return False
        
        recent_window = rewards[-window_size:]
        previous_window = rewards[-window_size*2:-window_size]
        
        # Check if recent performance is not significantly different
        recent_mean = np.mean(recent_window)
        previous_mean = np.mean(previous_window)
        
        improvement = (recent_mean - previous_mean) / abs(previous_mean) if previous_mean != 0 else 0
        return abs(improvement) < 0.05  # Less than 5% improvement
    
    def _analyze_exploration_decay(self) -> Dict[str, float]:
        """Analyze exploration decay patterns."""
        exploration_metrics = {}
        
        # Get exploration data from algorithm if available
        if hasattr(self.algorithm, 'agents'):
            for i, agent in enumerate(self.algorithm.agents):
                if hasattr(agent, 'epsilon'):
                    exploration_metrics[f'agent_{i}_epsilon'] = agent.epsilon
                elif hasattr(agent, 'exploration_noise'):
                    exploration_metrics[f'agent_{i}_noise'] = agent.exploration_noise
        
        return exploration_metrics
    
    def generate_research_report(self) -> Dict[str, Any]:
        """Generate a comprehensive research report."""
        report = {
            'experiment_info': {
                'algorithm': self.algorithm_name,
                'environment': self.env.__class__.__name__,
                'n_agents': self.n_agents,
                'device': str(self.device),
                'training_mode': self.training
            },
            'performance_metrics': self.get_research_metrics(),
            'training_summary': {
                'total_episodes': len(self.episode_rewards),
                'total_steps': self.total_steps,
                'training_time': getattr(self, 'training_time', 0),
                'best_performance': self.best_performance
            }
        }
        
        # Add algorithm-specific information
        research_interface = get_research_interface()
        algo_info = research_interface.discovery.get_algorithm_info(self.algorithm_name)
        if algo_info:
            report['algorithm_info'] = algo_info
        
        # Add configuration details
        report['configuration'] = {
            'hyperparameters': self._extract_hyperparameters(),
            'network_architecture': self._extract_network_info(),
            'training_parameters': self._extract_training_params()
        }
        
        return report
    
    def _extract_hyperparameters(self) -> Dict[str, Any]:
        """Extract current hyperparameters."""
        params = {}
        
        if hasattr(self.algorithm, 'agents') and self.algorithm.agents:
            agent = self.algorithm.agents[0]
            
            # Common hyperparameters
            if hasattr(agent, 'learning_rate'):
                params['learning_rate'] = agent.learning_rate
            if hasattr(agent, 'gamma'):
                params['gamma'] = agent.gamma
            if hasattr(agent, 'epsilon'):
                params['epsilon'] = agent.epsilon
            
            # Algorithm-specific parameters
            if hasattr(agent, 'clip_ratio'):
                params['clip_ratio'] = agent.clip_ratio
            if hasattr(agent, 'entropy_coef'):
                params['entropy_coef'] = agent.entropy_coef
        
        return params
    
    def _extract_network_info(self) -> Dict[str, Any]:
        """Extract network architecture information."""
        network_info = {}
        
        if hasattr(self.algorithm, 'agents') and self.algorithm.agents:
            agent = self.algorithm.agents[0]
            
            if hasattr(agent, 'policy_network'):
                network = agent.policy_network
                network_info['policy_network'] = self._describe_network(network)
            
            if hasattr(agent, 'value_network'):
                network = agent.value_network
                network_info['value_network'] = self._describe_network(network)
        
        return network_info
    
    def _describe_network(self, network) -> Dict[str, Any]:
        """Describe a neural network's architecture."""
        if network is None:
            return {}
        
        info = {
            'type': network.__class__.__name__,
            'parameters': sum(p.numel() for p in network.parameters() if p.requires_grad),
            'layers': []
        }
        
        # Add layer information if accessible
        for name, module in network.named_modules():
            if len(list(module.children())) == 0:  # Leaf module
                info['layers'].append({
                    'name': name,
                    'type': module.__class__.__name__
                })
        
        return info
    
    def _extract_training_params(self) -> Dict[str, Any]:
        """Extract training-related parameters."""
        return {
            'max_steps_per_episode': self.config.get('max_steps', 100),
            'log_interval': self.config.get('log_interval', 100),
            'save_interval': self.config.get('save_interval', 1000),
            'eval_interval': self.config.get('eval_interval', 500),
            'use_wandb': self.config.get('use_wandb', False),
            'seed': self.config.get('seed', 42)
        }
