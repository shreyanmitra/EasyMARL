import argparse
import random
import torch
import numpy as np
import wandb

import utils
from modern_multiagent_controller import ModernMultiAgentController
from algorithms import list_available_algorithms

def parse_args():
  parser = argparse.ArgumentParser(description='Multi-Agent Reinforcement Learning Framework')
  parser.add_argument(
      '--env_name', type=str, default='MultiGrid-Cluttered-Fixed-15x15',
      help='Name of environment.')
  parser.add_argument(
      '--algorithm', type=str, default='ippo',
      help="MARL algorithm to use. Options: ippo, maddpg, qmix, mappo")
  parser.add_argument(
      '--mode', type=str, default=None,
      help="Deprecated: use --algorithm instead. For backward compatibility.")
  parser.add_argument(
      '--with_expert', type=str, default=None,
      help="Whether to train with an expert")
  parser.add_argument(
      '--debug', action=argparse.BooleanOptionalAction,
      help="If used will disable wandb logging.")
  parser.add_argument(
      '--seed', type=int, default=None,
      help="Random seed.")
  parser.add_argument(
      '--keep_training', action=argparse.BooleanOptionalAction,
      help="If used will continue training from previous checkpoint.")
  parser.add_argument(
      '--visualize', action=argparse.BooleanOptionalAction,
      help="If used will run evaluation with visualization.")
  parser.add_argument(
      '--evaluate', action=argparse.BooleanOptionalAction,
      help="If used will run evaluation only.")
  parser.add_argument(
      '--video_dir', type=str, default='videos',
      help="Name of location to store videos.")
  parser.add_argument(
      '--load_checkpoint_from',  type=str, default=None,
      help="Path to find model checkpoints to load")
  parser.add_argument(
        '--wandb_project', type=str, default='MARL_Training',
        help="Name of wandb project.")
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
