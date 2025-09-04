# EasyMARL CLI Documentation

This document provides comprehensive documentation for the EasyMARL Command Line Interface (CLI).

## Overview

The EasyMARL CLI provides command-line access to all framework functionality, making it easy to train agents, manage environments, and run experiments without writing Python code.

## Installation

The CLI is automatically available after installing EasyMARL:

```bash
pip install easymarl
```

## Available Commands

### easymarl-train
Train MARL agents with specified algorithms.

```bash
# Basic usage
easymarl-train --algorithm qmix --env MultiGrid-Empty-6x6-v0

# Advanced usage
easymarl-train --algorithm ippo --vectorized --n_envs 8 --episodes 1000
```

**Arguments:**
- `--algorithm`: MARL algorithm (qmix, ippo, maddpg, mappo, etc.)
- `--env_name`: Environment name
- `--episodes`: Number of training episodes
- `--vectorized`: Use parallel environments
- `--n_envs`: Number of parallel environments
- `--evaluate`: Run evaluation only
- `--visualize`: Generate visualization videos
- `--debug`: Disable Weights & Biases logging
- `--seed`: Random seed for reproducibility

### easymarl-gui
Launch the graphical user interface.

```bash
# Launch Gradio interface (default)
easymarl-gui

# Launch with specific options
easymarl-gui --interface gradio --port 7860
```

### easymarl-demo
Run demonstration examples.

```bash
# Basic demo
easymarl-demo --example basic

# Quick demo with fewer episodes
easymarl-demo --example comparison --quick
```

### easymarl-wandb
Weights & Biases integration.

```bash
# Login to W&B
easymarl-wandb --login

# View project
easymarl-wandb --view --project EasyMARL

# Create sweep
easymarl-wandb --sweep config.yaml --project my_project
```

### easymarl-env
Environment management.

```bash
# List available environments
easymarl-env --list

# Get environment info
easymarl-env --info MultiGrid-Empty-6x6-v0

# Create custom environment
easymarl-env --create my_env --template cluttered --size 8x8
```

### easymarl-algo
Algorithm management and information.

```bash
# List all algorithms
easymarl-algo --list

# Get algorithm details
easymarl-algo --info qmix

# Show algorithm taxonomy
easymarl-algo --taxonomy

# Compare algorithms
easymarl-algo --compare qmix vdn ippo --env MultiGrid-Empty-6x6-v0
```

## Examples

### Quick Start Training
```bash
# Train IPPO agents (beginner-friendly)
easymarl-train --algorithm ippo --episodes 500

# Train QMIX agents with visualization
easymarl-train --algorithm qmix --visualize --episodes 1000

# Fast training with vectorization
easymarl-train --algorithm mappo --vectorized --n_envs 16
```

### Algorithm Comparison
```bash
# Compare value-based methods
easymarl-algo --compare qmix vdn qtran --episodes 200

# Run comprehensive benchmark
easymarl-algo --benchmark --env MultiGrid-Cluttered-Fixed-15x15
```

### Environment Exploration
```bash
# Explore available environments
easymarl-env --list

# Test environment functionality
easymarl-env --test MultiGrid-Dynamic-Obstacles-6x6-v0

# Create custom training environment
easymarl-env --create training_env --template empty --agents 4
```

### Experiment Management
```bash
# Set up W&B project
easymarl-wandb --login
easymarl-wandb --project my_research

# Run experiment with tracking
easymarl-train --algorithm maddpg --wandb_project my_research

# View results
easymarl-wandb --view --project my_research
```

## Configuration

CLI commands use the same configuration system as the Python API. Configuration files are located in:
- `config/default.yaml` - Default settings
- `config/mode/*.yaml` - Algorithm-specific configurations
- `config/domain/*.yaml` - Environment-specific configurations

## Tips and Best Practices

1. **Start Simple**: Begin with `easymarl-demo --example basic`
2. **Use Vectorization**: Add `--vectorized` for faster training
3. **Set Seeds**: Use `--seed` for reproducible experiments
4. **Monitor Progress**: Enable W&B with `--wandb_project`
5. **Quick Testing**: Use `--debug` to disable logging during development

## Troubleshooting

### Common Issues

**Command not found:**
```bash
# Ensure EasyMARL is installed
pip install easymarl

# Check installation
python -c "import easymarl; print('OK')"
```

**Environment errors:**
```bash
# List available environments
easymarl-env --list

# Validate environment
easymarl-env --validate MultiGrid-Empty-6x6-v0
```

**Algorithm errors:**
```bash
# Check available algorithms
easymarl-algo --list

# Get algorithm information
easymarl-algo --info qmix
```

For more help, see the main documentation at: https://shreyanmitra.github.io/EasyMARL
