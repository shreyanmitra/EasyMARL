# EasyMARL API Documentation

This document provides comprehensive API documentation for the EasyMARL framework.

## Core API

### UnifiedMultiAgentController

The main interface for training and evaluating MARL algorithms.

```python
from controllers.unified_multiagent_controller import UnifiedMultiAgentController
import gymnasium as gym

# Create environment
env = gym.make('MultiGrid-Empty-6x6-v0')

# Create controller
controller = UnifiedMultiAgentController(
    env=env,
    algorithm='qmix',
    educational_mode=True
)

# Train agents
controller.train(episodes=1000)

# Evaluate performance
results = controller.evaluate(episodes=100)
```

#### Parameters

- **env** (gym.Env): Multi-agent environment
- **algorithm** (str): MARL algorithm name ('qmix', 'ippo', 'maddpg', etc.)
- **educational_mode** (bool): Enable detailed explanations and logging
- **config** (dict, optional): Custom configuration overrides
- **seed** (int, optional): Random seed for reproducibility

#### Methods

##### train(episodes=1000, **kwargs)
Train the multi-agent system.

**Parameters:**
- `episodes` (int): Number of training episodes
- `save_interval` (int): Episodes between model saves
- `eval_interval` (int): Episodes between evaluations
- `log_interval` (int): Episodes between logging

**Returns:**
- `TrainingResults`: Training metrics and statistics

##### evaluate(episodes=100, render=False)
Evaluate trained agents.

**Parameters:**
- `episodes` (int): Number of evaluation episodes
- `render` (bool): Render episodes during evaluation

**Returns:**
- `EvaluationResults`: Performance metrics

##### save_models(path)
Save trained models to disk.

##### load_models(path)
Load trained models from disk.

### Algorithm Registry

Access available algorithms through the registry system.

```python
from algorithms import ALGORITHM_REGISTRY, get_algorithm_class

# List all algorithms
print(ALGORITHM_REGISTRY.keys())

# Get specific algorithm
QMIXAlgorithm = get_algorithm_class('QMIX')
```

### Environment Management

#### VectorizedEnvironment

Parallel environment execution for faster training.

```python
from environments.vectorized_env import VectorizedEnvironment

# Create vectorized environment
vec_env = VectorizedEnvironment(
    env_name='MultiGrid-Empty-6x6-v0',
    n_envs=8
)

# Use with controller
controller = UnifiedMultiAgentController(
    env=vec_env,
    algorithm='ippo'
)
```

#### Environment Creation

```python
import gymnasium as gym

# Standard environments
env = gym.make('MultiGrid-Empty-6x6-v0')
env = gym.make('MultiGrid-Cluttered-Fixed-15x15-v0')

# Custom environments
from environments.gym_multigrid import MultiGridEnv

env = MultiGridEnv(
    grid_size=10,
    num_agents=3,
    max_steps=250
)
```

### Configuration Management

#### ConfigManager

Centralized configuration management.

```python
from core.config_manager import ConfigManager

# Get configuration manager
config_manager = ConfigManager()

# Load algorithm configuration
qmix_config = config_manager.get_config('qmix')

# Load default configuration
default_config = config_manager.get_default_config()

# Create custom configuration
custom_config = config_manager.create_config(
    algorithm='mappo',
    overrides={
        'learning_rate': 0.0001,
        'batch_size': 64
    }
)
```

### Research Interface

Advanced research tools and experiment management.

```python
from core.research_interface import ResearchInterface

# Create research interface
research = ResearchInterface()

# Setup experiment
experiment = research.setup_experiment(
    name='multi_agent_cooperation',
    algorithms=['qmix', 'vdn', 'ippo'],
    environments=['MultiGrid-Empty-6x6-v0'],
    seeds=[42, 123, 456]
)

# Run experiment
results = research.run_experiment(experiment)

# Analyze results
analysis = research.analyze_results(results)
```

## Algorithm APIs

### Value-Based Algorithms

#### QMIX

```python
from algorithms.model_free.value_based.approximation.qmix import QMIX

# Create QMIX algorithm
qmix = QMIX(
    state_dim=observation_space.shape[0],
    action_dim=action_space.n,
    num_agents=env.num_agents,
    learning_rate=0.0005
)

# Training step
loss = qmix.train_step(batch)

# Action selection
actions = qmix.select_actions(observations)
```

#### VDN

```python
from algorithms.model_free.value_based.approximation.vdn import VDN

vdn = VDN(
    state_dim=observation_space.shape[0],
    action_dim=action_space.n,
    num_agents=env.num_agents
)
```

### Policy-Based Algorithms

#### IPPO

```python
from algorithms.model_free.policy_based.discrete_action.ippo import IPPO

ippo = IPPO(
    observation_space=env.observation_space,
    action_space=env.action_space,
    num_agents=env.num_agents,
    learning_rate=0.0003
)
```

#### MAPPO

```python
from algorithms.model_free.policy_based.discrete_action.mappo import MAPPO

mappo = MAPPO(
    observation_space=env.observation_space,
    action_space=env.action_space,
    num_agents=env.num_agents,
    centralized_critic=True
)
```

### Actor-Critic Algorithms

#### MADDPG

```python
from algorithms.model_free.actor_critic.maddpg import MADDPG

maddpg = MADDPG(
    observation_spaces=env.observation_spaces,
    action_spaces=env.action_spaces,
    num_agents=env.num_agents,
    actor_lr=0.001,
    critic_lr=0.001
)
```

#### COMA

```python
from algorithms.model_free.actor_critic.coma import COMA

coma = COMA(
    observation_space=env.observation_space,
    action_space=env.action_space,
    num_agents=env.num_agents,
    centralized_critic=True
)
```

## Utility APIs

### Performance Utilities

```python
from core.utils.advanced import enable_jit_compilation, setup_performance_monitoring

# Enable JIT compilation
enable_jit_compilation()

# Setup performance monitoring
metrics = setup_performance_monitoring()
```

### Logging and Visualization

```python
from core.utils.enhanced import setup_advanced_monitoring

# Setup monitoring
monitor = setup_advanced_monitoring(
    wandb_project='my_project',
    log_interval=100
)

# Log metrics
monitor.log_metrics({
    'episode_reward': 150.0,
    'episode_length': 200,
    'loss': 0.05
})
```

## GUI APIs

### Gradio Interface

```python
from gui.gradio_interface import create_interface, launch_gui

# Create interface
interface = create_interface()

# Launch GUI
launch_gui(port=7860, share=False)
```

### React Backend API

```python
from api.flask_backend import app

# Run Flask backend
app.run(host='0.0.0.0', port=5000, debug=False)
```

## Network APIs

### Custom Networks

```python
from networks.custom_networks import create_network

# Create custom network
network = create_network(
    network_type='attention',
    input_dim=64,
    hidden_dims=[128, 128],
    output_dim=32
)
```

### MultiGrid Networks

```python
from networks.multigrid_network import MultiGridNetwork

# Specialized network for MultiGrid environments
network = MultiGridNetwork(
    observation_space=env.observation_space,
    action_space=env.action_space,
    num_agents=env.num_agents
)
```

## Error Handling

### Common Exceptions

```python
from easymarl.exceptions import (
    EasyMARLError,
    AlgorithmError,
    EnvironmentError,
    ConfigurationError
)

try:
    controller.train(episodes=1000)
except AlgorithmError as e:
    print(f"Algorithm error: {e}")
except EnvironmentError as e:
    print(f"Environment error: {e}")
except ConfigurationError as e:
    print(f"Configuration error: {e}")
```

## Examples

### Basic Training Example

```python
import gymnasium as gym
from controllers.unified_multiagent_controller import UnifiedMultiAgentController

# Setup
env = gym.make('MultiGrid-Empty-6x6-v0')
controller = UnifiedMultiAgentController(
    env=env,
    algorithm='qmix',
    educational_mode=True
)

# Train
results = controller.train(episodes=1000)
print(f"Average reward: {results.avg_reward:.2f}")

# Evaluate
eval_results = controller.evaluate(episodes=100)
print(f"Evaluation reward: {eval_results.avg_reward:.2f}")
```

### Advanced Configuration Example

```python
from core.config_manager import ConfigManager
from controllers.unified_multiagent_controller import UnifiedMultiAgentController

# Custom configuration
config_manager = ConfigManager()
custom_config = config_manager.create_config(
    algorithm='maddpg',
    overrides={
        'learning_rate': 0.0001,
        'batch_size': 128,
        'replay_buffer_size': 100000,
        'target_update_interval': 100
    }
)

# Create controller with custom config
controller = UnifiedMultiAgentController(
    env=env,
    algorithm='maddpg',
    config=custom_config
)
```

### Research Experiment Example

```python
from core.research_interface import ResearchInterface

# Setup research interface
research = ResearchInterface()

# Define experiment
experiment_config = {
    'name': 'coordination_study',
    'algorithms': ['qmix', 'vdn', 'ippo', 'mappo'],
    'environments': [
        'MultiGrid-Empty-6x6-v0',
        'MultiGrid-Cluttered-Fixed-15x15-v0'
    ],
    'seeds': [42, 123, 456, 789, 999],
    'episodes': 2000,
    'evaluations': 5
}

# Run experiment
results = research.run_experiment(experiment_config)

# Generate report
report = research.generate_report(results)
print(report.summary)
```

For more examples and tutorials, see: https://shreyanmitra.github.io/EasyMARL
