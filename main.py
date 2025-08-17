"""
(C) Shreyan Mitra, based on starter code by Natasha Jaques

EasyMARL - Multi-Agent Reinforcement Learning Framework
Main Entry Point for Training and Evaluation

This is the primary script for running MARL experiments. It provides a simple
command-line interface for training and evaluating different MARL algorithms
on various multi-agent environments.

For MARL Beginners:
This is your starting point! Run this script to train agents using different
algorithms. Start with simple commands like:
  python main.py --algorithm ippo --env_name MultiGrid-Cluttered-Fixed-15x15

Key Features:
- Support for 20+ MARL algorithms (IPPO, QMIX, MADDPG, MAPPO, etc.)
- Multiple environments (MultiGrid, custom environments)
- Experiment tracking with Weights & Biases
- Automatic checkpointing and model saving
- Evaluation and visualization modes
- Reproducible experiments with seed setting

Usage Examples:
  # Train IPPO agents (good for beginners)
  python main.py --algorithm ippo
  
  # Train QMIX agents with specific environment
  python main.py --algorithm qmix --env_name MultiGrid-Cluttered-Fixed-15x15
  
  # Evaluate trained models with visualization
  python main.py --evaluate --visualize --algorithm ippo
  
  # Continue training from checkpoint
  python main.py --algorithm ippo --keep_training
"""

# Import essential libraries
import argparse    # For parsing command-line arguments
import random      # For random number generation
import torch       # PyTorch for deep learning
import numpy as np # Numerical computations
import wandb       # Weights & Biases for experiment tracking

# Import our framework components
import utils                                        # Utility functions
from modern_multiagent_controller import ModernMultiAgentController  # Main training controller
from algorithms import list_available_algorithms   # Available MARL algorithms

def parse_args():
    """
    Parse command-line arguments for configuring MARL experiments.
    
    This function defines all the options you can specify when running the script,
    such as which algorithm to use, which environment to train on, etc.
    
    Returns:
        argparse.Namespace: Parsed arguments with all configuration options
    
    For Beginners:
    These are all the "settings" you can adjust when running experiments.
    Most have sensible defaults, so you can start with just specifying the algorithm.
    """
    parser = argparse.ArgumentParser(
        description='EasyMARL - Multi-Agent Reinforcement Learning Framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train IPPO agents (recommended for beginners)
  python main.py --algorithm ippo
  
  # Train QMIX agents on specific environment
  python main.py --algorithm qmix --env_name MultiGrid-Cluttered-Fixed-15x15
  
  # Evaluate trained models with visualization
  python main.py --evaluate --visualize --algorithm ippo
  
  # List all available algorithms
  python main.py --list_algorithms
        """
    )
    
    # Core Configuration Options
    parser.add_argument(
        '--env_name', type=str, default='MultiGrid-Cluttered-Fixed-15x15',
        help='Environment to train on. Examples: MultiGrid-Cluttered-Fixed-15x15, MultiGrid-Empty-8x8')
    
    parser.add_argument(
        '--algorithm', type=str, default='ippo',
        help='MARL algorithm to use. Popular options: ippo (beginner-friendly), qmix, maddpg, mappo')
    
    # Legacy support for backward compatibility
    parser.add_argument(
        '--mode', type=str, default=None,
        help='Deprecated: use --algorithm instead. Kept for backward compatibility.')
    
    # Advanced Training Options
    parser.add_argument(
        '--with_expert', type=str, default=None,
        help='Train with an expert agent (advanced feature)')
    
    # Debugging and Development Options
    parser.add_argument(
        '--debug', action=argparse.BooleanOptionalAction,
        help='Disable wandb logging for local debugging')
    
    parser.add_argument(
        '--seed', type=int, default=None,
        help='Random seed for reproducible experiments (recommended: 42, 123, 456)')
    
    # Training Control Options
    parser.add_argument(
        '--keep_training', action=argparse.BooleanOptionalAction,
        help='Continue training from the most recent checkpoint')
    
    # Evaluation and Analysis Options
    parser.add_argument(
        '--visualize', action=argparse.BooleanOptionalAction,
        help='Run with visualization (great for seeing what agents learned)')
    
    parser.add_argument(
        '--evaluate', action=argparse.BooleanOptionalAction,
        help='Run evaluation only (no training)')
    
    # Output and Storage Options
    parser.add_argument(
        '--video_dir', type=str, default='videos',
        help='Directory to save evaluation videos')
    
    parser.add_argument(
        '--load_checkpoint_from', type=str, default=None,
        help='Specific checkpoint path to load models from')
    
    # Experiment Tracking Options
    parser.add_argument(
        '--wandb_project', type=str, default='MARL_Training',
        help='Weights & Biases project name for experiment tracking')
    
    parser.add_argument(
        '--list_algorithms', action=argparse.BooleanOptionalAction,
        help="List available algorithms and exit.")

  return parser.parse_args()

def get_controller_class(config):
    return ModernMultiAgentController

def initialize(algorithm, env_name, debug, visualize, evaluate, seed, with_expert, wandb_project):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Handle backward compatibility
    if algorithm is None:
        algorithm = 'ippo'  # Default algorithm
    
    # Determine mode for config generation (backward compatibility)
    mode = 'ppo' if algorithm in ['ippo', 'ppo'] else algorithm

    config = utils.generate_parameters(
      mode=mode, domain=env_name, debug=(debug or visualize or evaluate),
      seed=seed, with_expert=with_expert, wandb_project=wandb_project)

    # Add algorithm name to config
    config.algorithm = algorithm

    # Set seeds
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    env = utils.make_env(config)

    controller_class = get_controller_class(config)

    return device, config, env, controller_class

def main(args):
    # Handle special commands
    if args.list_algorithms:
        list_available_algorithms()
        return

    # Handle backward compatibility
    algorithm = args.algorithm
    if args.mode and not algorithm:
        algorithm = 'ippo' if args.mode.lower() == 'ppo' else args.mode.lower()
    
    device, config, env, controller_class = initialize(
      algorithm, args.env_name, args.debug, args.visualize, args.evaluate, 
      args.seed, args.with_expert, args.wandb_project)

    # Ensure if you're logging to wandb, it's to the right wandb
    if not args.debug and not args.visualize and not args.evaluate:  # Real run that logs to wandb
      if not args.wandb_project:
        print('ERROR: when logging to wandb, must specify a valid wandb project.')
        exit(1)

    # Create controller
    training_mode = not (args.visualize or args.evaluate)
    controller = controller_class(
        env=env, 
        config=config, 
        device=device, 
        algorithm=algorithm,
        training=training_mode,
        debug=args.debug
    )

    # Load models if specified
    if args.load_checkpoint_from:
        controller.load_models(args.load_checkpoint_from)

    if args.visualize:
        print('Generating visualization...')
        controller.visualize_episode(0)
        print(f'A video of the trained policies being tested in the environment '
              f'has been generated and is located in {args.video_dir}')
        return

    if args.evaluate:
        print('Running evaluation...')
        eval_metrics = controller.evaluate(num_episodes=20, render=False)
        print('Evaluation Results:')
        for key, value in eval_metrics.items():
            print(f'  {key}: {value:.4f}')
        return

    # Train Model
    print(f'Starting training with {algorithm.upper()} algorithm...')
    controller.train(config.n_episodes)
    
    # Print final statistics
    stats = controller.get_statistics()
    print('\nTraining completed!')
    print('Final Statistics:')
    for key, value in stats.items():
        print(f'  {key}: {value}')

if __name__ == '__main__':
    args = parse_args()
    main(args)
    main(parse_args())
