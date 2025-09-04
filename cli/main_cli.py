#!/usr/bin/env python3
"""
Main CLI for EasyMARL Training

This module provides the primary command-line interface for training MARL agents.
It wraps the functionality in main.py with a clean CLI interface.

Usage:
    easymarl-train --algorithm qmix --env MultiGrid-Empty-6x6-v0 --episodes 1000
    easymarl-gui  # Launch graphical interface
    easymarl-demo # Run demonstration examples
"""

import argparse
import sys
import os

# Add the parent directory to Python path to import EasyMARL modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from main import parse_args, main as main_func
    from algorithms import list_available_algorithms
    from core.utils.base import get_project_root
except ImportError as e:
    print(f"❌ Error importing EasyMARL modules: {e}")
    print("💡 Make sure you're running from the EasyMARL directory")
    sys.exit(1)

def create_train_parser():
    """Create argument parser for training command."""
    parser = argparse.ArgumentParser(
        description="EasyMARL Training CLI - Train Multi-Agent Reinforcement Learning algorithms",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  easymarl-train --algorithm qmix --env MultiGrid-Empty-6x6-v0
  easymarl-train --algorithm ippo --vectorized --n_envs 8
  easymarl-train --algorithm maddpg --episodes 1000 --seed 42
  easymarl-train --list_algorithms
        """
    )
    
    # Algorithm selection
    parser.add_argument('--algorithm', type=str, default='ippo',
                       help='MARL algorithm to use (default: ippo)')
    
    # Environment configuration
    parser.add_argument('--env_name', type=str, default='MultiGrid-Cluttered-Fixed-15x15',
                       help='Environment name (default: MultiGrid-Cluttered-Fixed-15x15)')
    
    # Training configuration
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of training episodes (default: 1000)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    # Execution modes
    parser.add_argument('--evaluate', action='store_true',
                       help='Run evaluation only (no training)')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualization videos')
    parser.add_argument('--keep_training', action='store_true',
                       help='Continue training from existing checkpoint')
    
    # Performance options
    parser.add_argument('--vectorized', action='store_true',
                       help='Use vectorized environments for faster training')
    parser.add_argument('--n_envs', type=int, default=8,
                       help='Number of parallel environments (default: 8)')
    
    # Logging and debugging
    parser.add_argument('--debug', action='store_true',
                       help='Disable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='EasyMARL',
                       help='Weights & Biases project name (default: EasyMARL)')
    
    # Information commands
    parser.add_argument('--list_algorithms', action='store_true',
                       help='List all available algorithms and exit')
    
    return parser

def create_gui_parser():
    """Create argument parser for GUI command."""
    parser = argparse.ArgumentParser(
        description="EasyMARL GUI - Launch graphical user interface",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--interface', type=str, choices=['gradio', 'react'], default='gradio',
                       help='GUI interface type (default: gradio)')
    parser.add_argument('--port', type=int, default=None,
                       help='Port for web interface (default: auto)')
    parser.add_argument('--host', type=str, default='localhost',
                       help='Host for web interface (default: localhost)')
    
    return parser

def create_demo_parser():
    """Create argument parser for demo command."""
    parser = argparse.ArgumentParser(
        description="EasyMARL Demo - Run demonstration examples",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--example', type=str, default='basic',
                       choices=['basic', 'advanced', 'comparison', 'custom'],
                       help='Demo example to run (default: basic)')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick demo with fewer episodes')
    
    return parser

def train_command(args):
    """Execute training command."""
    if args.list_algorithms:
        print("🧠 Available MARL Algorithms:")
        print("=" * 50)
        algorithms = list_available_algorithms()
        for category, algs in algorithms.items():
            print(f"\n{category.upper()}:")
            for alg in algs:
                print(f"  • {alg}")
        return 0
    
    print("🚀 Starting EasyMARL Training...")
    print(f"   Algorithm: {args.algorithm}")
    print(f"   Environment: {args.env_name}")
    print(f"   Episodes: {args.episodes}")
    
    try:
        # Use the main function from main.py
        main_func(args)
        print("✅ Training completed successfully!")
        return 0
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return 1

def gui_command(args):
    """Execute GUI command."""
    print("🌐 Launching EasyMARL GUI...")
    
    try:
        if args.interface == 'gradio':
            import easymarl
            easymarl.launch_gui()
        elif args.interface == 'react':
            print("🚀 Starting React interface...")
            print("   Frontend: http://localhost:3000")
            print("   Backend: http://localhost:5000/api")
            print("💡 Use './tools/start-easymarl.sh' for full React setup")
            
        return 0
    except Exception as e:
        print(f"❌ GUI launch failed: {e}")
        return 1

def demo_command(args):
    """Execute demo command."""
    print(f"🎮 Running EasyMARL Demo: {args.example}")
    
    try:
        if args.example == 'basic':
            # Import and run basic demo
            from examples.demo import run_basic_demo
            run_basic_demo(quick=args.quick)
        elif args.example == 'comparison':
            from examples.demo import run_algorithm_comparison
            run_algorithm_comparison(quick=args.quick)
        else:
            print(f"💡 Demo '{args.example}' not yet implemented")
            print("   Available demos: basic, comparison")
            
        return 0
    except ImportError:
        print("💡 Running basic training demo...")
        # Fallback to basic training
        demo_args = argparse.Namespace(
            algorithm='ippo',
            env_name='MultiGrid-Empty-6x6-v0',
            episodes=50 if args.quick else 200,
            evaluate=False,
            visualize=True,
            debug=True,
            vectorized=False,
            n_envs=1,
            seed=42,
            keep_training=False,
            wandb_project='EasyMARL-Demo',
            list_algorithms=False
        )
        return train_command(demo_args)
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return 1

def main():
    """Main CLI entry point."""
    # Check if this is a specific command
    if len(sys.argv) > 0:
        command = os.path.basename(sys.argv[0])
        
        if 'easymarl-gui' in command:
            parser = create_gui_parser()
            args = parser.parse_args()
            return gui_command(args)
        elif 'easymarl-demo' in command:
            parser = create_demo_parser()
            args = parser.parse_args()
            return demo_command(args)
        elif 'easymarl-train' in command or 'easymarl' in command:
            parser = create_train_parser()
            args = parser.parse_args()
            return train_command(args)
    
    # Default to training interface
    parser = create_train_parser()
    args = parser.parse_args()
    return train_command(args)

if __name__ == '__main__':
    sys.exit(main())
