#!/usr/bin/env python3
"""
(C) Shreyan Mitra, based on starter code by Natasha Jaques

EasyMARL Framework Demonstration Script
Interactive Examples for Learning Multi-Agent Reinforcement Learning

This script provides hands-on demonstrations of different MARL algorithms,
showing how to train agents and compare their performance. Perfect for
beginners who want to see MARL algorithms in action!

For MARL Beginners:
This is your playground! Run this script to see different MARL algorithms
train and compare their performance. It's designed to be educational and
help you understand how different algorithms behave.

Demo Features:
- Algorithm Comparison: Train multiple algorithms on the same task
- Performance Visualization: See training curves and final performance
- Quick Training: Shorter episodes for fast demonstration
- Educational Output: Explanations of what's happening during training

Usage Examples:
  python demo.py                    # Run default algorithm comparison
  python demo.py --algorithm ippo   # Demo specific algorithm
  python demo.py --episodes 100     # Adjust training length
  python demo.py --env_name MultiGrid-Empty-6x6  # Try different environment

What You'll Learn:
- How different MARL algorithms train
- Performance differences between algorithms  
- How training progresses over time
- Which algorithms work well for different tasks

Recommended Learning Path:
1. Start with IPPO (simplest multi-agent extension of single-agent learning)
2. Try QMIX (value-based cooperation through mixing networks)
3. Experiment with MAPPO (state-of-the-art policy gradient method)
4. Compare results to understand algorithm differences
"""

# Import essential libraries
import torch                # PyTorch for deep learning
import matplotlib.pyplot as plt  # For plotting training curves
import numpy as np          # Numerical computations
import argparse            # Command-line argument parsing
import os                  # Operating system interface
import time                # Time measurement

# Import EasyMARL framework components
from algorithms import create_marl_algorithm, list_available_algorithms  # Algorithm factory
from modern_multiagent_controller import ModernMultiAgentController      # Training controller
import utils               # Utility functions


def demo_algorithm_comparison():
    """
    Compare different MARL algorithms on the same environment.
    
    This function trains multiple algorithms on the same task and compares
    their learning curves. It's designed to be educational and help beginners
    understand the differences between MARL algorithms.
    
    For MARL Beginners:
    This is like a race between different AI training methods! You'll see
    which algorithms learn faster, achieve better performance, or are more
    stable during training.
    """
    print("🚀 EasyMARL Algorithm Comparison Demo")
    print("=" * 60)
    print("Training multiple MARL algorithms and comparing their performance...")
    print("This demo will help you understand how different algorithms behave!")
    
    # Demo Configuration
    # Choose a moderately challenging environment for meaningful comparison
    env_name = 'MultiGrid-Cluttered-Fixed-15x15'
    
    # Select algorithms that represent different approaches to MARL
    algorithms = ['ippo', 'mappo', 'qmix']  # Subset for demo
    print(f"\nAlgorithms to compare: {', '.join([alg.upper() for alg in algorithms])}")
    print("- IPPO: Independent agents (simplest approach)")
    print("- MAPPO: Centralized training with cooperation") 
    print("- QMIX: Value-based mixing networks")
    
    # Keep training short for demonstration purposes
    episodes_per_algorithm = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTraining {episodes_per_algorithm} episodes per algorithm on {device}")
    
    results = {}  # Store training results for comparison
    
    # Train each algorithm and collect results
    for algorithm in algorithms:
        print(f"\n🔄 Training {algorithm.upper()} for {episodes_per_algorithm} episodes...")
        print(f"   Algorithm Type: {_get_algorithm_description(algorithm)}")
        
        # Setup configuration for current algorithm
        mode = 'ppo' if algorithm in ['ippo', 'ppo'] else algorithm
        config = utils.generate_parameters(
            mode=mode,
            domain=env_name,
            debug=True,  # Disable wandb for demo to keep output clean
            seed=42,     # Fixed seed for fair comparison
            with_expert=None,
            wandb_project='demo_project'
        )
        
        # Override settings for demonstration
        config.n_episodes = episodes_per_algorithm        # Short training for demo
        config.log_interval = 10                          # Frequent progress updates
        config.save_interval = episodes_per_algorithm + 1 # Don't save models during demo
        config.visualize_every = episodes_per_algorithm + 1  # Don't visualize during demo
        
        # Create environment and controller
        env = utils.make_env(config)
        controller = ModernMultiAgentController(
            env=env,
            config=config,
            device=device,
            algorithm=algorithm,
            training=True,
            debug=True
        )
        
        # Train
        start_time = time.time()
        controller.train(episodes_per_algorithm)
        training_time = time.time() - start_time
        
        # Get statistics
        stats = controller.get_statistics()
        results[algorithm] = {
            'rewards': controller.episode_rewards,
            'lengths': controller.episode_lengths,
            'final_performance': stats['mean_reward'],
            'training_time': training_time
        }
        
        print(f"✅ {algorithm.upper()} completed in {training_time:.2f}s")
        print(f"   Final average reward: {stats['mean_reward']:.2f}")
    
    # Plot comparison
    plot_comparison(results, episodes_per_algorithm)
    
    # Print summary
    print_summary(results)


def plot_comparison(results, episodes):
    """Plot learning curves for different algorithms."""
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Learning curves
    plt.subplot(1, 3, 1)
    for algorithm, data in results.items():
        # Smooth the learning curve
        rewards = data['rewards']
        if len(rewards) > 10:
            window = min(10, len(rewards) // 5)
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            x = np.arange(len(smoothed))
        else:
            smoothed = rewards
            x = np.arange(len(smoothed))
        
        plt.plot(x, smoothed, label=algorithm.upper(), linewidth=2)
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Final performance comparison
    plt.subplot(1, 3, 2)
    algorithms = list(results.keys())
    final_rewards = [results[alg]['final_performance'] for alg in algorithms]
    
    bars = plt.bar(algorithms, final_rewards, color=['skyblue', 'lightcoral', 'lightgreen'])
    plt.ylabel('Average Reward')
    plt.title('Final Performance')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, final_rewards):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.2f}', ha='center', va='bottom')
    
    # Plot 3: Training time comparison
    plt.subplot(1, 3, 3)
    training_times = [results[alg]['training_time'] for alg in algorithms]
    
    bars = plt.bar(algorithms, training_times, color=['gold', 'salmon', 'lightseagreen'])
    plt.ylabel('Training Time (seconds)')
    plt.title('Training Efficiency')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, training_times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{value:.1f}s', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs('demo_results', exist_ok=True)
    plt.savefig('demo_results/algorithm_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Comparison plot saved to demo_results/algorithm_comparison.png")


def print_summary(results):
    """Print a summary of the comparison."""
    print(f"\n📋 COMPARISON SUMMARY")
    print("=" * 60)
    
    # Find best performing algorithm
    best_alg = max(results.keys(), key=lambda x: results[x]['final_performance'])
    fastest_alg = min(results.keys(), key=lambda x: results[x]['training_time'])
    
    print(f"🏆 Best Performance: {best_alg.upper()} "
          f"({results[best_alg]['final_performance']:.2f} average reward)")
    print(f"⚡ Fastest Training: {fastest_alg.upper()} "
          f"({results[fastest_alg]['training_time']:.2f}s)")
    
    print(f"\nDetailed Results:")
    print("-" * 40)
    for algorithm, data in results.items():
        print(f"{algorithm.upper():<8} | "
              f"Reward: {data['final_performance']:6.2f} | "
              f"Time: {data['training_time']:6.2f}s")


def demo_single_algorithm():
    """Demonstrate training a single algorithm with visualization."""
    print("🎯 Single Algorithm Training Demo")
    print("=" * 60)
    
    algorithm = 'ippo'
    episodes = 100
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = utils.generate_parameters(
        mode='ppo',
        domain='MultiGrid-Cluttered-Fixed-15x15',
        debug=True,
        seed=42,
        with_expert=None,
        wandb_project='demo_project'
    )
    
    config.n_episodes = episodes
    config.log_interval = 20
    config.save_interval = 50
    config.visualize_every = 50
    
    # Create environment and controller
    env = utils.make_env(config)
    controller = ModernMultiAgentController(
        env=env,
        config=config,
        device=device,
        algorithm=algorithm,
        training=True,
        debug=True
    )
    
    print(f"Training {algorithm.upper()} for {episodes} episodes...")
    
    # Train
    controller.train(episodes)
    
    # Evaluate
    print("\nRunning evaluation...")
    eval_metrics = controller.evaluate(num_episodes=10)
    
    print("Evaluation Results:")
    for key, value in eval_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Generate visualization
    print("\nGenerating visualization...")
    controller.visualize_episode(0)
    
    # Show final statistics
    stats = controller.get_statistics()
    print(f"\nFinal Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


def demo_algorithm_features():
    """Demonstrate specific features of different algorithms."""
    print("🔧 Algorithm Features Demo")
    print("=" * 60)
    
    algorithms_info = {
        'ippo': "Independent learning with PPO - good for simple cooperation",
        'maddpg': "Centralized training for non-stationary environments",
        'qmix': "Value decomposition for cooperative tasks",
        'mappo': "Centralized critics with decentralized policies"
    }
    
    print("Available Algorithms and Their Features:")
    print("-" * 50)
    
    for alg, description in algorithms_info.items():
        print(f"• {alg.upper():<8}: {description}")
    
    print(f"\nFor detailed algorithm information, run:")
    print(f"python main.py --list_algorithms")


def main():
    parser = argparse.ArgumentParser(description='EasyMARL Framework Demo')
    parser.add_argument('--demo', type=str, default='comparison',
                      choices=['comparison', 'single', 'features'],
                      help='Type of demo to run')
    
    args = parser.parse_args()
    
    print("🌟 Welcome to EasyMARL Demo!")
    print(f"Using device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    
    if args.demo == 'comparison':
        demo_algorithm_comparison()
    elif args.demo == 'single':
        demo_single_algorithm()
    elif args.demo == 'features':
        demo_algorithm_features()
    
    print(f"\n🎉 Demo completed! Check out the full framework features:")
    print(f"   • python main.py --help")
    print(f"   • python main.py --list_algorithms")
    print(f"   • python test_framework.py")


if __name__ == '__main__':
    main()
