"""
(C) Shreyan Mitra, based on starter code by Natasha Jaques

Enhanced Web-Based GUI for EasyMARL Framework

This module provides a professional, web-based graphical interface for training
and evaluating multi-agent reinforcement learning algorithms. It makes MARL
accessible to both beginners and researchers through an intuitive interface.

Key Features for Users:
🎯 Point-and-Click Training: No command-line experience required
📊 Real-Time Visualization: Live training graphs and performance metrics
🧠 Algorithm Explorer: Detailed descriptions of 21+ MARL algorithms
🎮 Environment Previews: Visual representations of training environments
📈 Experiment Tracking: Integration with Weights & Biases
💾 Data Export: Download training results and configurations
🎛️ Dual Controllers: Choose between Simple (educational) and Modern (production)

Interface Design:
1. 📋 Algorithm Selection Tab: Choose and learn about algorithms
2. ⚙️ Configuration Tab: Set training parameters and options
3. 🚀 Training Tab: Start training and monitor progress
4. 📊 Results Tab: Analyze performance and download data
5. 🎥 Visualization Tab: Watch trained agents in action

For MARL Beginners:
This GUI removes all technical barriers to MARL experimentation. You can:
- Learn about different algorithms through interactive descriptions
- Train agents with just a few clicks
- See real-time progress without reading logs
- Understand what your agents learned through visualizations

For MARL Researchers:
This GUI provides a rapid prototyping environment where you can:
- Quickly test different algorithm configurations
- Compare multiple approaches side-by-side
- Export results for publication
- Share experiments with collaborators

Technical Architecture:
- Frontend: Gradio web interface (automatic responsive design)
- Backend: Python with asyncio for concurrent training
- Data Flow: WebSocket-like updates for real-time visualization
- Integration: Seamless connection to EasyMARL training pipeline

Usage:
1. Run: python gui.py
2. Open browser to displayed URL
3. Select algorithm and configure parameters
4. Click "Start Training" and watch real-time progress
5. Evaluate and visualize results
"""

# =============================================================================
# IMPORTS: External libraries and internal framework components
# =============================================================================

import gradio as gr          # Modern web-based GUI framework for ML applications
import argparse             # Command-line argument parsing for legacy compatibility
import random               # Random number generation for reproducible experiments
import torch                # PyTorch deep learning framework
import numpy as np          # Numerical computing library
import wandb                # Weights & Biases experiment tracking platform
import yaml                 # YAML configuration file parser
import os                   # Operating system interface for file operations
import matplotlib.pyplot as plt  # Plotting library for training visualizations
import matplotlib           # Matplotlib configuration and backend settings
matplotlib.use('Agg')       # Use non-interactive backend for server/headless environments
from threading import Thread    # Multi-threading for concurrent training execution
import time                 # Time utilities for timestamps and scheduling
from typing import Dict, List, Tuple  # Type hints for better code documentation
import json                 # JSON serialization for data export and configuration

# EasyMARL Framework Components
import utils                # Core utility functions (environment creation, config management)
from modern_multiagent_controller import ModernMultiAgentController    # Production-ready training controller
from simple_multiagent_controller import SimpleMultiAgentController    # Educational training controller

# Enhanced features
from utils import (
    ENHANCED_FEATURES_AVAILABLE, ADVANCED_FEATURES_AVAILABLE,
    setup_world_class_training, make_production_vec_env
)

# =============================================================================
# GLOBAL CONFIGURATION: Environment and Algorithm Specifications
# =============================================================================

# Educational mode setting - can be modified at runtime
EDUCATIONAL_MODE = True

# Supported Multi-Agent Environments
# These environments are available for training through the GUI interface
AVAILABLE_ENVS = ["MultiGrid-Cluttered-Fixed-15x15"]

# List of all 21+ available MARL algorithms
# Each algorithm represents a different approach to multi-agent learning
AVAILABLE_ALGORITHMS = [
    "ippo",        # Independent Proximal Policy Optimization
    "maddpg",      # Multi-Agent Deep Deterministic Policy Gradient
    "qmix",        # Q-Mix Value Decomposition
    "mappo",       # Multi-Agent Proximal Policy Optimization
    "iql",         # Independent Q-Learning
    "vdn",         # Value Decomposition Networks
    "coma",        # Counterfactual Multi-Agent Policy Gradients
    "qtran",       # Q-Transformation
    "maven",       # Multi-Agent Variational Exploration
    "hql",         # Hysteretic Q-Learning
    "lql",         # Lenient Q-Learning
    "wolfphc",     # Win-or-Learn-Fast Policy Hill Climbing
    "nashq",       # Nash-Q Learning
    "dcg",         # Deep Coordination Graphs
    "minimaxq",    # Minimax-Q Learning
    "maacc",       # Multi-Agent Actor-Critic with Counterfactual Critic
    "nfsp",        # Neural Fictitious Self-Play
    "mfq",         # Mean Field Q-Learning
    "maddpgcomm",  # MADDPG with Communication
    "comacomm"     # COMA with Communication
]

# =============================================================================
# ALGORITHM DESCRIPTIONS: Comprehensive information for each algorithm
# =============================================================================

# Dictionary containing detailed descriptions for each algorithm
# This helps users understand which algorithm to choose for their task
ALGORITHM_DESCRIPTIONS = {
    "ippo": {
        "name": "Independent Proximal Policy Optimization",
        "type": "On-policy, Decentralized",
        "description": "Each agent learns independently using PPO with GAE. Simple, stable, works well for fully cooperative tasks with simple coordination needs.",
        "features": ["GAE (Generalized Advantage Estimation)", "PPO clipping", "Entropy regularization", "Independent learning"],
        "best_for": "Fully cooperative tasks, simple coordination scenarios",
        "pros": ["Simple and stable", "Works well in practice", "Good baseline"],
        "cons": ["No explicit coordination", "May struggle with complex interactions"]
    },
    "maddpg": {
        "name": "Multi-Agent Deep Deterministic Policy Gradient", 
        "type": "Off-policy, Centralized Training",
        "description": "Centralized training with decentralized execution. Each agent has a centralized critic that can access global information during training.",
        "features": ["Centralized critics", "Decentralized actors", "Experience replay", "Soft target updates"],
        "best_for": "Mixed-motive scenarios, continuous/discrete actions",
        "pros": ["Handles non-stationary environments", "Stable learning"],
        "cons": ["Can be sample inefficient", "Requires careful tuning"]
    },
    "qmix": {
        "name": "Q-Mix Value Decomposition",
        "type": "Off-policy, Value-based",
        "description": "Learns individual Q-functions for each agent and a mixing network that combines them while maintaining the Individual-Global-Max (IGM) principle.",
        "features": ["Mixing networks", "Individual-Global-Max principle", "Value decomposition", "Centralized training"],
        "best_for": "Fully cooperative tasks, discrete actions",
        "pros": ["Principled credit assignment", "Theoretical guarantees"],
        "cons": ["Limited to cooperative settings", "Monotonicity constraint"]
    },
    "mappo": {
        "name": "Multi-Agent Proximal Policy Optimization",
        "type": "On-policy, Centralized Training",
        "description": "Multi-agent extension of PPO with centralized value functions. Combines benefits of PPO with centralized training for better coordination.",
        "features": ["Centralized value functions", "Parameter sharing", "Centralized training", "Decentralized execution"],
        "best_for": "Fully cooperative tasks, complex coordination",
        "pros": ["Combines PPO benefits with centralization", "Good performance"],
        "cons": ["Computationally expensive", "Memory intensive"]
    },
    "iql": {
        "name": "Independent Q-Learning",
        "type": "Off-policy, Independent",
        "description": "Simple baseline where each agent learns independently using Q-learning. No coordination between agents.",
        "features": ["Epsilon-greedy exploration", "Q-table or neural networks", "Independent learning"],
        "best_for": "Simple environments, baseline comparisons",
        "pros": ["Simple implementation", "Fast training", "Good baseline"],
        "cons": ["No coordination", "Struggles with agent interactions"]
    },
    "vdn": {
        "name": "Value Decomposition Networks",
        "type": "Off-policy, Value-based",
        "description": "Learns individual value functions and combines them additively. Simpler than QMIX but with stronger assumptions.",
        "features": ["Additive value decomposition", "Individual Q-networks", "Centralized training"],
        "best_for": "Cooperative tasks with additive rewards",
        "pros": ["Simple and interpretable", "Fast convergence"],
        "cons": ["Strong additivity assumption", "Limited expressiveness"]
    },
    "coma": {
        "name": "Counterfactual Multi-Agent Policy Gradients",
        "type": "On-policy, Actor-Critic",
        "description": "Uses counterfactual reasoning for credit assignment. Centralized critic estimates counterfactual baselines for each agent's actions.",
        "features": ["Counterfactual reasoning", "Centralized critic", "Credit assignment", "Policy gradients"],
        "best_for": "Cooperative tasks requiring credit assignment",
        "pros": ["Principled credit assignment", "Handles reward sparsity"],
        "cons": ["High variance", "Computational complexity"]
    },
    "qtran": {
        "name": "Q-Transformation",
        "type": "Off-policy, Value-based", 
        "description": "Relaxes the monotonicity constraint of QMIX using regularization terms. More expressive than QMIX for complex coordination.",
        "features": ["Relaxed monotonicity", "Regularization terms", "Value decomposition", "Transformation networks"],
        "best_for": "Complex cooperative coordination tasks",
        "pros": ["More expressive than QMIX", "Better coordination"],
        "cons": ["More complex", "Harder to tune"]
    },
    "maven": {
        "name": "Multi-Agent Variational Exploration",
        "type": "On-policy, Exploration-Enhanced",
        "description": "Adds hierarchical exploration using a latent variable that encourages diverse trajectories and coordinated exploration.",
        "features": ["Latent diversity", "Hierarchical exploration", "Mutual information", "Coordinated exploration"],
        "best_for": "Exploration-heavy environments, diverse behaviors",
        "pros": ["Structured exploration", "Diverse behaviors", "Better sample efficiency"],
        "cons": ["Additional complexity", "Hyperparameter sensitive"]
    },
    "hql": {
        "name": "Hysteretic Q-Learning",
        "type": "Off-policy, Exploration-Enhanced",
        "description": "Uses different learning rates for positive and negative TD errors to handle non-stationarity in multi-agent environments.",
        "features": ["Hysteretic updates", "Non-stationarity handling", "Asymmetric learning rates"],
        "best_for": "Non-stationary environments, competitive scenarios",
        "pros": ["Handles non-stationarity", "Robust to other learning agents"],
        "cons": ["Hyperparameter tuning", "May slow convergence"]
    },
    "lql": {
        "name": "Lenient Q-Learning",
        "type": "Off-policy, Exploration-Enhanced",
        "description": "Uses temperature-based leniency that is more forgiving of potentially sub-optimal joint actions during exploration.",
        "features": ["Lenient updates", "Temperature decay", "Exploration tolerance", "Maximum Q-value tracking"],
        "best_for": "Environments with exploration challenges",
        "pros": ["Forgives exploration mistakes", "Better coordination"],
        "cons": ["Parameter sensitive", "May delay convergence"]
    },
    "wolfphc": {
        "name": "Win-or-Learn-Fast Policy Hill Climbing",
        "type": "Game-theoretic, Self-Play",
        "description": "Adapts learning rate based on whether the agent is winning or losing. Fast adaptation when losing, slow when winning.",
        "features": ["Adaptive learning rates", "Win/loss detection", "Policy hill climbing", "Game-theoretic learning"],
        "best_for": "Competitive scenarios, game-theoretic settings",
        "pros": ["Adaptive to performance", "Game-theoretic foundation"],
        "cons": ["Win/loss definition required", "Complex implementation"]
    },
    "nashq": {
        "name": "Nash-Q Learning",
        "type": "Game-theoretic, Equilibrium-based",
        "description": "Computes Nash equilibrium at each state using linear programming. Provides theoretical guarantees for convergence.",
        "features": ["Nash equilibrium computation", "Linear programming", "Game-theoretic optimality", "Multi-agent Q-learning"],
        "best_for": "Small state spaces, theoretical analysis",
        "pros": ["Theoretical guarantees", "Game-theoretic optimality"],
        "cons": ["Computational complexity", "Scalability issues"]
    },
    "dcg": {
        "name": "Deep Coordination Graphs",
        "type": "Value-based, Graph-based",
        "description": "Uses coordination graphs to model agent interactions and performs message passing for coordination. Scalable coordination mechanism.",
        "features": ["Coordination graphs", "Message passing", "Payoff functions", "Graph neural networks"],
        "best_for": "Large-scale coordination, structured interactions",
        "pros": ["Scalable coordination", "Structured interactions", "Efficient coordination"],
        "cons": ["Graph structure required", "Message passing overhead"]
    },
    "minimaxq": {
        "name": "Minimax-Q Learning",
        "type": "Game-theoretic, Competitive",
        "description": "Minimax approach for two-agent zero-sum games. Computes minimax optimal policies assuming rational opponents.",
        "features": ["Minimax optimization", "Zero-sum games", "Security strategies", "Game-theoretic learning"],
        "best_for": "Two-agent zero-sum games, competitive scenarios",
        "pros": ["Optimal for zero-sum games", "Security guarantees"],
        "cons": ["Limited to zero-sum", "Two-agent restriction"]
    },
    "maacc": {
        "name": "Multi-Agent Actor-Critic with Counterfactual Critic",
        "type": "Actor-Critic, Centralized Training",
        "description": "Actor-critic method with counterfactual critics for improved credit assignment and coordination in cooperative settings.",
        "features": ["Counterfactual critics", "Actor-critic architecture", "Credit assignment", "Centralized training"],
        "best_for": "Cooperative tasks requiring credit assignment",
        "pros": ["Good credit assignment", "Stable learning"],
        "cons": ["Computational overhead", "Parameter tuning"]
    },
    "nfsp": {
        "name": "Neural Fictitious Self-Play",
        "type": "Game-theoretic, Self-Play",
        "description": "Combines reinforcement learning with supervised learning. Learns best response and average strategy networks for Nash equilibrium.",
        "features": ["Dual networks", "Self-play", "Supervised learning", "Nash equilibrium", "Reservoir sampling"],
        "best_for": "Game-theoretic scenarios, competitive settings",
        "pros": ["Nash equilibrium convergence", "Game-theoretic foundation"],
        "cons": ["Complex implementation", "Dual network overhead"]
    },
    "mfq": {
        "name": "Mean Field Q-Learning",
        "type": "Large-scale, Population-based",
        "description": "Approximates large-scale multi-agent interactions using mean field theory. Scales to hundreds of agents.",
        "features": ["Mean field approximation", "Population dynamics", "Large-scale scalability", "Interaction modeling"],
        "best_for": "Large-scale environments, hundreds of agents",
        "pros": ["Highly scalable", "Population-level modeling"],
        "cons": ["Mean field assumptions", "Homogeneous agents"]
    },
    "maddpgcomm": {
        "name": "MADDPG with Communication",
        "type": "Communication-based, Centralized Training",
        "description": "Extends MADDPG with explicit communication channels. Agents learn to generate and use messages for coordination.",
        "features": ["Communication channels", "Message generation", "Centralized training", "Explicit coordination"],
        "best_for": "Tasks requiring explicit communication",
        "pros": ["Explicit communication", "Better coordination"],
        "cons": ["Communication overhead", "Message interpretation"]
    },
    "comacomm": {
        "name": "COMA with Communication",
        "type": "Communication-based, Actor-Critic",
        "description": "Combines COMA's counterfactual reasoning with communication mechanisms for improved coordination and credit assignment.",
        "features": ["Communication", "Counterfactual reasoning", "Credit assignment", "Message passing"],
        "best_for": "Cooperative tasks with communication needs",
        "pros": ["Communication + credit assignment", "Coordinated exploration"],
        "cons": ["High complexity", "Communication costs"]
    }
}

# =============================================================================
# GLOBAL STATE VARIABLES: Real-time training data management
# =============================================================================

# Global variables for GUI state management
currentEnv = None  # Stores the current environment instance for visualization

# Global dictionary to store real-time training data
# This data is shared between the background training thread and GUI components
current_training_data = {
    "episode_rewards": [],    # List of rewards for each completed episode
    "episode_lengths": [],    # List of episode lengths (number of steps)
    "episodes": [],           # List of episode numbers for x-axis plotting
    "is_training": False,     # Boolean flag indicating if training is active
    "algorithm": "",          # Name of currently selected algorithm
    "env_name": ""           # Name of currently selected environment
}

# =============================================================================
# VISUALIZATION FUNCTIONS: Plot creation and data visualization
# =============================================================================

def create_training_plot():
    """
    Create and return a matplotlib figure showing training progress.
    
    This function generates a two-panel plot:
    - Top panel: Episode rewards over time with moving average
    - Bottom panel: Episode lengths over time with moving average
    
    Returns:
        matplotlib.figure.Figure: The generated plot figure
    """
    # Check if we have enough data points to create a meaningful plot
    if len(current_training_data["episode_rewards"]) < 2:
        # Create empty plot with proper labels when no data is available
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        ax1.set_title("Episode Rewards")     # Title for rewards subplot
        ax1.set_xlabel("Episode")            # X-axis label
        ax1.set_ylabel("Reward")             # Y-axis label
        ax2.set_title("Episode Lengths")     # Title for lengths subplot
        ax2.set_xlabel("Episode")            # X-axis label
        ax2.set_ylabel("Length")             # Y-axis label
        plt.tight_layout()                   # Adjust spacing between subplots
        return fig
    
    # Create plot with actual training data
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Extract data from global training state
    episodes = current_training_data["episodes"]        # Episode numbers for x-axis
    rewards = current_training_data["episode_rewards"]  # Reward values for y-axis
    lengths = current_training_data["episode_lengths"]  # Length values for y-axis
    
    # Plot episode rewards in the top subplot
    ax1.plot(episodes, rewards, 'b-', alpha=0.6, label='Episode Reward')
    
    # Add moving average if we have enough data points
    if len(rewards) > 10:
        # Calculate window size for moving average (max 50, min 25% of data)
        window = min(50, len(rewards) // 4)
        # Compute moving average using convolution for efficiency
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        # Plot moving average line (offset x-axis by window size)
        ax1.plot(episodes[window-1:], moving_avg, 'r-', linewidth=2, label=f'Moving Avg ({window})')
    
    # Configure the rewards subplot
    ax1.set_title(f"Training Progress - {current_training_data['algorithm'].upper()} on {current_training_data['env_name']}")
    ax1.set_xlabel("Episode")                # X-axis label
    ax1.set_ylabel("Reward")                 # Y-axis label
    ax1.legend()                             # Show legend with line labels
    ax1.grid(True, alpha=0.3)                # Add light grid for readability
    
    # Plot episode lengths in the bottom subplot
    ax2.plot(episodes, lengths, 'g-', alpha=0.6, label='Episode Length')
    
    # Add moving average for lengths if we have enough data
    if len(lengths) > 10:
        window = min(50, len(lengths) // 4)
        moving_avg = np.convolve(lengths, np.ones(window)/window, mode='valid')
        ax2.plot(episodes[window-1:], moving_avg, 'orange', linewidth=2, label=f'Moving Avg ({window})')
    
    # Configure the lengths subplot
    ax2.set_xlabel("Episode")                # X-axis label
    ax2.set_ylabel("Length")                 # Y-axis label
    ax2.legend()                             # Show legend
    ax2.grid(True, alpha=0.3)                # Add grid
    
    plt.tight_layout()                       # Adjust layout to prevent overlap
    return fig

def get_algorithm_info(algorithm):
    """Get detailed algorithm information."""
    if algorithm not in ALGORITHM_DESCRIPTIONS:
        return "Algorithm information not available."
    
    info = ALGORITHM_DESCRIPTIONS[algorithm]
    
    description = f"""
## {info['name']} ({algorithm.upper()})

**Type:** {info['type']}

**Description:** {info['description']}

**Best For:** {info['best_for']}

### Key Features:
{chr(10).join([f"• {feature}" for feature in info['features']])}

### Pros:
{chr(10).join([f"✅ {pro}" for pro in info['pros']])}

### Cons:
{chr(10).join([f"❌ {con}" for con in info['cons']])}
"""
    
    return description

def get_environment_grid():
    """Get the current environment visualization."""
    if currentEnv is None:
        return None
    try:
        return currentEnv.render('rgb_array')
    except:
        return None

def update_training_data(episode, reward, length):
    """Update training data for plotting."""
    current_training_data["episodes"].append(episode)
    current_training_data["episode_rewards"].append(reward)
    current_training_data["episode_lengths"].append(length)

class TrainingThread(Thread):
    """Thread for running training in background."""
    
    def __init__(self, env_name, algorithm, config, device, controller_type="simple", use_enhanced_features=False):
        super().__init__()
        self.env_name = env_name
        self.algorithm = algorithm
        self.config = config
        self.device = device
        self.controller_type = controller_type  # New parameter for controller selection
        self.use_enhanced_features = use_enhanced_features  # Enhanced features flag
        self.status = "Starting..."
        self.error = None
        self.daemon = True
        
    def run(self):
        try:
            # Reset training data
            current_training_data["episode_rewards"] = []
            current_training_data["episode_lengths"] = []
            current_training_data["episodes"] = []
            current_training_data["is_training"] = True
            current_training_data["algorithm"] = self.algorithm
            current_training_data["env_name"] = self.env_name
            
            global currentEnv
            currentEnv = utils.make_env(self.env_name)
            
            # Create controller based on selected type
            if self.controller_type == "simple":
                # Use Simple Controller - beginner-friendly structure with optional enhancements
                controller = SimpleMultiAgentController(
                    env=currentEnv,
                    config=self.config,
                    device=self.device,
                    algorithm=self.algorithm,
                    training=True,
                    use_enhanced_features=self.use_enhanced_features and ENHANCED_FEATURES_AVAILABLE
                )
            else:
                # Use Modern Controller - advanced features with enhancements
                experiment_name = f"{self.algorithm}_{self.env_name}_{int(time.time())}"
                
                controller = ModernMultiAgentController(
                    env=currentEnv,
                    config=self.config,
                    device=self.device,
                    algorithm=self.algorithm,
                    training=True,
                    experiment_name=experiment_name,
                    enable_advanced_tracking=self.use_enhanced_features and ADVANCED_FEATURES_AVAILABLE,
                    enable_performance_monitoring=True
                )
            
            # Start wandb logging
            if self.config.get('use_wandb', False):
                wandb.init(
                    project=self.config.get('wandb_project', 'easymarl'),
                    name=f"{self.algorithm}_{self.env_name}",
                    config=self.config
                )
            
            self.status = "Training in progress..."
            
            # Custom training loop with GUI updates
            total_episodes = self.config.get('max_episodes', 1000)
            for episode in range(total_episodes):
                # Run one episode
                episode_data = self.run_one_episode(controller, episode)
                
                # Update GUI data
                update_training_data(
                    episode, 
                    episode_data['total_reward'], 
                    episode_data['episode_length']
                )
                
                # Update status
                if episode % 100 == 0:
                    avg_reward = np.mean(current_training_data["episode_rewards"][-100:])
                    self.status = f"Episode {episode}/{total_episodes} | Avg Reward: {avg_reward:.2f}"
                
                # Break if requested
                if not current_training_data["is_training"]:
                    break
            
            current_training_data["is_training"] = False
            self.status = f"Training completed! Final average reward: {np.mean(current_training_data['episode_rewards'][-100:]):.2f}"
            
            # Save models
            controller.save_models("final")
            
            if wandb.run is not None:
                wandb.finish()
                
        except Exception as e:
            self.error = str(e)
            self.status = f"Training failed: {str(e)}"
            current_training_data["is_training"] = False
    
    def run_one_episode(self, controller, episode):
        """Run a single episode and return metrics."""
        obs = controller.env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        episode_rewards = []
        
        while not done and episode_length < controller.config.get('max_steps', 100):
            # Get actions
            actions = controller._get_actions(obs, training=True)
            
            # Take step
            next_obs, rewards, done, info = controller.env.step(actions)
            
            # Process rewards
            if isinstance(rewards, list):
                step_reward = sum(rewards)
                episode_rewards.append(rewards)
            else:
                step_reward = rewards
                episode_rewards.append([rewards] * controller.n_agents)
            
            episode_reward += step_reward
            episode_length += 1
            obs = next_obs
            
            # Store transitions in agent memories (for algorithms that need it)
            for i, agent in enumerate(controller.algorithm.agents):
                if hasattr(agent, 'store_transition'):
                    agent_obs = controller._extract_agent_obs(obs, i)
                    agent_reward = rewards[i] if isinstance(rewards, list) else rewards
                    agent.store_transition(
                        agent_obs, actions[i], 0.0, 0.0, agent_reward, done
                    )
        
        # Update models
        if hasattr(controller.algorithm, 'train_step'):
            rollout_data = {
                'episode_rewards': episode_rewards,
                'episode_length': episode_length,
                'total_reward': episode_reward
            }
            metrics = controller.algorithm.train_step(rollout_data)
        
        # Log to wandb
        if wandb.run is not None:
            wandb.log({
                'episode/reward': episode_reward,
                'episode/length': episode_length,
                'episode/x_axis': episode
            })
        
        return {
            'total_reward': episode_reward,
            'episode_length': episode_length,
            'episode_rewards': episode_rewards
        }

training_thread = None

def buttonClicked(env_name, algorithm, max_episodes, use_wandb, learning_rate, controller_type, use_enhanced_features=False):
    """Train the selected algorithm on the selected environment."""
    global training_thread
    
    # Stop existing training if running
    if training_thread and training_thread.is_alive():
        current_training_data["is_training"] = False
        return "Stopping previous training..."
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load algorithm configuration
    config_path = f"config/mode/{algorithm}.yaml"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Fallback to default config
        config = {
            'algorithm': algorithm,
            'max_episodes': max_episodes,
            'max_steps': 100,
            'learning_rate': learning_rate,
            'gamma': 0.99,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.995,
            'batch_size': 32,
            'memory_size': 10000,
            'hidden_dim': 128,
            'use_wandb': use_wandb,
            'log_interval': 10,
            'save_interval': 500,
            'eval_interval': 250
        }
    
    # Override config with GUI settings
    config['max_episodes'] = max_episodes
    config['use_wandb'] = use_wandb
    config['learning_rate'] = learning_rate
    config['environment'] = env_name
    config['device'] = str(device)
    
    # Set random seeds
    seed = config.get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Start training in background thread
    training_thread = TrainingThread(env_name, algorithm, config, device, controller_type, use_enhanced_features)
    training_thread.start()
    
    enhanced_status = " with Enhanced Features" if use_enhanced_features and ENHANCED_FEATURES_AVAILABLE else ""
    controller_name = "Simple" if controller_type == "simple" else "Modern"
    return f"Training started for {algorithm.upper()} on {env_name} using {controller_name} Controller{enhanced_status}. Check the training graph for progress!"

def stop_training():
    """Stop the current training."""
    global training_thread
    if training_thread and training_thread.is_alive():
        current_training_data["is_training"] = False
        return "Stopping training..."
    else:
        return "No training in progress."

def get_training_status():
    """Get current training status."""
    global training_thread
    if training_thread and training_thread.is_alive():
        return training_thread.status
    elif current_training_data["is_training"]:
        return "Training in progress..."
    else:
        return "Ready to start training."

def download_training_data():
    """Prepare training data for download."""
    if not current_training_data["episode_rewards"]:
        return None
    
    # Create downloadable data
    data = {
        "algorithm": current_training_data["algorithm"],
        "environment": current_training_data["env_name"],
        "episodes": current_training_data["episodes"],
        "rewards": current_training_data["episode_rewards"],
        "lengths": current_training_data["episode_lengths"]
    }
    
    # Save to file
    filename = f"training_data_{current_training_data['algorithm']}_{int(time.time())}.json"
    filepath = os.path.join("outputs", filename)
    os.makedirs("outputs", exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    return filepath

def evaluateModel(env_name, algorithm, model_path):
    """Evaluate a trained model."""
    if not os.path.exists(model_path):
        return f"Model path {model_path} does not exist"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load configuration
    config_path = f"config/mode/{algorithm}.yaml"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        return f"Configuration for {algorithm} not found"
    
    config['environment'] = env_name
    config['device'] = str(device)
    
    # Create environment
    env = utils.make_env(env_name)
    
    # Create controller with enhanced features for evaluation
    experiment_name = f"eval_{algorithm}_{env_name}_{int(time.time())}"
    
    controller = ModernMultiAgentController(
        env=env,
        config=config,
        device=device,
        algorithm=algorithm,
        training=False,
        experiment_name=experiment_name,
        enable_advanced_tracking=False,  # Disable for evaluation
        enable_performance_monitoring=True
    )
    
    try:
        # Load model and evaluate
        controller.load_models(model_path)
        results = controller.evaluate(num_episodes=10)
        
        result_str = "## Evaluation Results\n\n"
        for key, value in results.items():
            if isinstance(value, float):
                result_str += f"**{key.replace('_', ' ').title()}:** {value:.4f}\n\n"
            else:
                result_str += f"**{key.replace('_', ' ').title()}:** {value}\n\n"
        
        return result_str
    except Exception as e:
        return f"Evaluation failed: {str(e)}"

# Update functions for real-time updates
def update_plot():
    """Update the training plot."""
    return create_training_plot()

def update_status():
    """Update training status."""
    return get_training_status()

def update_env_grid():
    """Update environment visualization."""
    return get_environment_grid()

def on_algorithm_change(algorithm):
    """Handle algorithm selection change."""
    return get_algorithm_info(algorithm)


with gr.Blocks(title="EasyMARL Framework", theme=gr.themes.Soft()) as interface:
    gr.Markdown("# 🚀 EasyMARL - Comprehensive Multi-Agent Reinforcement Learning Framework")
    gr.Markdown("Train and evaluate 21+ state-of-the-art MARL algorithms on multi-agent environments!")
    
    with gr.Tabs():
        # Training Tab
        with gr.TabItem("🎯 Training", id="training"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Configuration")
                    env_dropdown = gr.Dropdown(
                        label="Environment", 
                        choices=AVAILABLE_ENVS,
                        value=AVAILABLE_ENVS[0]
                    )
                    algorithm_dropdown = gr.Dropdown(
                        label="Algorithm", 
                        choices=AVAILABLE_ALGORITHMS,
                        value="qmix"
                    )
                    
                    # Controller selection - Choose between simple and modern controllers
                    controller_type = gr.Dropdown(
                        label="Controller Type",
                        choices=[
                            ("Simple (Beginner-friendly)", "simple"),
                            ("Modern (Advanced features)", "modern")
                        ],
                        value="simple",
                        info="Simple: Structured like original metacontroller, easier to understand. Modern: Full-featured with advanced metrics."
                    )
                    
                    # Training parameters
                    with gr.Row():
                        max_episodes = gr.Number(
                            label="Max Episodes",
                            value=1000,
                            minimum=10,
                            maximum=10000
                        )
                        learning_rate = gr.Number(
                            label="Learning Rate",
                            value=0.001,
                            minimum=1e-6,
                            maximum=1.0
                        )
                    
                    use_wandb = gr.Checkbox(
                        label="Enable WandB Logging",
                        value=False
                    )
                    
                    # Enhanced features option (if available)
                    use_enhanced_features = gr.Checkbox(
                        label="Enable Enhanced Features (10x Performance)",
                        value=ENHANCED_FEATURES_AVAILABLE,
                        interactive=ENHANCED_FEATURES_AVAILABLE,
                        info="Enhanced vectorization, JIT compilation, and world-class optimizations. Only available if enhanced features are installed."
                    )
                    
                    with gr.Row():
                        train_button = gr.Button("🚀 Start Training", variant="primary", size="lg")
                        stop_button = gr.Button("⏹️ Stop Training", variant="stop")
                    
                    training_status = gr.Textbox(
                        label="Training Status", 
                        value="Ready to start training.",
                        interactive=False
                    )
                    
                with gr.Column(scale=2):
                    gr.Markdown("### Training Progress")
                    
                    # Real-time training plot
                    training_plot = gr.Plot(
                        label="Training Metrics",
                        value=create_training_plot()
                    )
                    
                    # Download button
                    download_button = gr.Button("💾 Download Training Data", size="sm")
                    download_file = gr.File(label="Training Data", visible=False)
                    
                    # Environment visualization
                    env_image = gr.Image(
                        label="Environment State",
                        width=300,
                        height=300
                    )
            
            # Algorithm description (updates based on selection)
            algorithm_info = gr.Markdown(
                value=get_algorithm_info("qmix"),
                label="Algorithm Information"
            )
            
            # Event handlers for training tab
            train_button.click(
                buttonClicked,
                inputs=[env_dropdown, algorithm_dropdown, max_episodes, use_wandb, learning_rate, controller_type, use_enhanced_features],
                outputs=training_status
            )
            
            stop_button.click(
                stop_training,
                outputs=training_status
            )
            
            algorithm_dropdown.change(
                on_algorithm_change,
                inputs=algorithm_dropdown,
                outputs=algorithm_info
            )
            
            download_button.click(
                download_training_data,
                outputs=download_file
            )
            
            # Auto-update components every 2 seconds during training
            interface.load(
                update_plot,
                outputs=training_plot,
                every=2
            )
            
            interface.load(
                update_status,
                outputs=training_status,
                every=1
            )
            
            interface.load(
                update_env_grid,
                outputs=env_image,
                every=3
            )
        
        # Evaluation Tab
        with gr.TabItem("📊 Evaluation", id="evaluation"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Model Evaluation")
                    eval_env_dropdown = gr.Dropdown(
                        label="Environment", 
                        choices=AVAILABLE_ENVS,
                        value=AVAILABLE_ENVS[0]
                    )
                    eval_algorithm_dropdown = gr.Dropdown(
                        label="Algorithm", 
                        choices=AVAILABLE_ALGORITHMS,
                        value="qmix"
                    )
                    model_path_input = gr.Textbox(
                        label="Model Path",
                        placeholder="models/experiment/qmix_final",
                        info="Path to the trained model files"
                    )
                    eval_button = gr.Button("🔍 Evaluate Model", variant="secondary", size="lg")
                    
                with gr.Column():
                    eval_output = gr.Markdown(
                        value="### Evaluation Results\nSelect a model and click 'Evaluate Model' to see results.",
                        label="Results"
                    )
            
            eval_button.click(
                evaluateModel,
                inputs=[eval_env_dropdown, eval_algorithm_dropdown, model_path_input],
                outputs=eval_output
            )
        
        # Algorithm Information Tab
        with gr.TabItem("📚 Algorithm Catalog", id="algorithms"):
            gr.Markdown("## 🎯 Available MARL Algorithms")
            gr.Markdown("Comprehensive catalog of 21+ state-of-the-art multi-agent reinforcement learning algorithms.")
            
            # Algorithm selection for detailed info
            info_algorithm_dropdown = gr.Dropdown(
                label="Select Algorithm for Details",
                choices=AVAILABLE_ALGORITHMS,
                value="qmix"
            )
            
            detailed_algorithm_info = gr.Markdown(
                value=get_algorithm_info("qmix")
            )
            
            info_algorithm_dropdown.change(
                on_algorithm_change,
                inputs=info_algorithm_dropdown,
                outputs=detailed_algorithm_info
            )
            
            # Algorithm categories
            with gr.Accordion("📋 Algorithm Categories", open=True):
                gr.Markdown("""
                ### 🎯 Value Decomposition Methods
                **QMIX, VDN, QTRAN** - Decompose team value function into individual components
                
                ### 🎭 Actor-Critic Methods  
                **IPPO, MAPPO, MADDPG, COMA, MAACC** - Policy gradient methods with value function baselines
                
                ### 🎲 Independent Learning
                **IQL** - Simple baseline where agents learn independently
                
                ### 🎪 Exploration-Enhanced Methods
                **MAVEN, HQL, LQL** - Advanced exploration strategies for multi-agent environments
                
                ### 🎯 Game-Theoretic Methods
                **WoLF-PHC, Nash-Q, Minimax-Q, NFSP** - Game theory and equilibrium-based approaches
                
                ### 🌐 Large-Scale Methods
                **DCG, MFQ** - Coordination graphs and mean field approaches for scalability
                
                ### 📡 Communication-Based Methods
                **MADDPG-Comm, COMA-Comm** - Explicit communication protocols
                """)
        
        # Environment Information Tab
        with gr.TabItem("🌍 Environments", id="environments"):
            gr.Markdown("## 🌍 Multi-Agent Environments")
            
            with gr.Accordion("MultiGrid Environments", open=True):
                gr.Markdown("""
                ### MultiGrid-Cluttered-Fixed-15x15
                **Type:** Fully cooperative navigation
                
                **Description:** Agents must navigate through a cluttered 15x15 grid world to reach their goals while avoiding obstacles and coordinating with other agents.
                
                **Features:**
                - Fixed obstacle layout for consistent evaluation
                - Partial observability (7x7 view for each agent)
                - Discrete action space (6 actions: forward, left, right, pickup, drop, toggle)
                - Sparse rewards (goal reaching)
                - Variable number of agents (2-4 supported)
                
                **Challenges:**
                - Coordination to avoid collisions
                - Efficient path planning
                - Partial observability
                - Multi-agent credit assignment
                """)
        
        # Help & Documentation Tab
        with gr.TabItem("❓ Help", id="help"):
            gr.Markdown("""
            ## 🚀 How to Use EasyMARL GUI
            
            ### 🎯 Training Process
            1. **Select Environment**: Choose from available multi-agent environments
            2. **Choose Algorithm**: Pick from 21+ state-of-the-art MARL algorithms  
            3. **Select Controller Type**: 
               - **Simple Controller**: Beginner-friendly, structured like original metacontroller
               - **Modern Controller**: Advanced features with comprehensive metrics
            4. **Configure Parameters**: Set episodes, learning rate, and enable logging
            5. **Start Training**: Click "Start Training" and monitor progress in real-time
            6. **Monitor Progress**: Watch training graphs and environment visualization
            7. **Download Data**: Save training metrics for analysis
            
            ### 📊 Real-Time Features
            - **Live Training Graphs**: Episode rewards and lengths updated every 2 seconds
            - **Environment Visualization**: See current agent positions and environment state
            - **Training Status**: Real-time updates on training progress
            - **WandB Integration**: Professional experiment tracking and logging
            
            ### 🔍 Evaluation
            - Load trained models for evaluation
            - Comprehensive metrics including success rates
            - Compare different algorithms and configurations
            
            ### 💡 Tips for Best Results
            - Start with **QMIX** for cooperative tasks
            - Use **MADDPG** for mixed-motive scenarios  
            - Try **IPPO** as a simple baseline
            - Enable WandB logging for detailed analysis
            - Adjust learning rates based on algorithm (typically 1e-3 to 1e-4)
            - Use 1000+ episodes for stable results
            
            ### 🐛 Troubleshooting
            - Ensure PyTorch is installed with CUDA support for GPU training
            - Check that environment names match exactly
            - Model paths should point to saved checkpoint files
            - Stop current training before starting new experiments
            """)
    
    gr.Markdown("---")
    gr.Markdown("**🏆 EasyMARL Framework** - Comprehensive MARL with 21+ algorithms | Built with modern architecture for research and education")
    gr.Markdown("💫 **Features**: Real-time training graphs, WandB integration, comprehensive algorithm catalog, downloadable results")

def main(port=7860, share=False, educational_mode=True):
    """
    Main function to launch the Gradio interface.
    
    Args:
        port (int): Port to run the interface on
        share (bool): Whether to create a shareable public link
        educational_mode (bool): Enable educational explanations
    """
    # Set educational mode globally if needed
    global EDUCATIONAL_MODE
    EDUCATIONAL_MODE = educational_mode
    
    print(f"🎓 EasyMARL Gradio Interface")
    print(f"📡 Server starting on port {port}")
    if educational_mode:
        print("🎓 Educational mode: Detailed explanations enabled")
    
    interface.launch(
        server_port=port,
        share=share, 
        debug=False,
        show_error=True,
        quiet=False
    )

# Default launch for direct script execution
if __name__ == "__main__":
    main()
