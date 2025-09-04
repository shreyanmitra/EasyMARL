"""
CLI Module for EasyMARL

This module provides command-line interfaces for the EasyMARL framework,
making it easy to train agents, manage environments, and run experiments
from the command line.

Available CLI commands:
- easymarl-train: Train MARL agents with specified algorithms
- easymarl-gui: Launch the graphical user interface
- easymarl-demo: Run demonstration examples
- easymarl-wandb: Weights & Biases integration
- easymarl-env: Environment management
- easymarl-algo: Algorithm management and information
"""

from .main_cli import main as cli_main
from .wandb_cli import main as wandb_main
from .env_cli import main as env_main
from .algo_cli import main as algo_main

__all__ = ['cli_main', 'wandb_main', 'env_main', 'algo_main']
