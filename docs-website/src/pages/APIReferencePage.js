import React, { useState } from 'react';
import { 
  Search, 
  ChevronDown, 
  ChevronRight, 
  Code, 
  Settings, 
  Zap, 
  Users,
  Play,
  Type,
  ArrowRight,
  CheckCircle
} from 'lucide-react';
import CodeBlock from '../components/CodeBlock';

const APIReferencePage = () => {
  const [searchTerm, setSearchTerm] = useState('');
  const [expandedSections, setExpandedSections] = useState({
    'unified-controller': true,
    'configuration': false,
    'algorithms': false,
    'environments': false,
    'utilities': false
  });
  const [selectedMethod, setSelectedMethod] = useState(null);

  const toggleSection = (sectionId) => {
    setExpandedSections(prev => ({
      ...prev,
      [sectionId]: !prev[sectionId]
    }));
  };

  const apiSections = [
    {
      id: 'unified-controller',
      title: 'UnifiedMultiAgentController',
      icon: <Settings className="w-5 h-5" />,
      description: 'Main controller class for all multi-agent training and evaluation',
      methods: [
        {
          name: '__init__',
          signature: '__init__(env_name, algorithm, n_envs=1, educational_mode=False, **kwargs)',
          description: 'Initialize the unified multi-agent controller',
          parameters: [
            { name: 'env_name', type: 'str', description: 'Name of the environment to use' },
            { name: 'algorithm', type: 'str', description: 'Algorithm to use (ippo, qmix, maddpg, etc.)' },
            { name: 'n_envs', type: 'int', description: 'Number of parallel environments (default: 1)' },
            { name: 'educational_mode', type: 'bool', description: 'Enable detailed explanations (default: False)' },
            { name: 'config', type: 'dict', description: 'Algorithm-specific configuration parameters' }
          ],
          returns: { type: 'UnifiedMultiAgentController', description: 'Initialized controller instance' },
          example: `from easymarl.controllers import UnifiedMultiAgentController

# Basic initialization
controller = UnifiedMultiAgentController(
    env_name='MultiGrid-Empty-6x6',
    algorithm='ippo',
    n_envs=8,
    educational_mode=True
)

# Advanced configuration
controller = UnifiedMultiAgentController(
    env_name='MultiGrid-Complex-12x12',
    algorithm='qmix',
    n_envs=16,
    config={
        'learning_rate': 3e-4,
        'batch_size': 256,
        'mixer_hidden_dim': 64,
        'exploration_epsilon_start': 1.0,
        'exploration_epsilon_end': 0.05
    }
)`
        },
        {
          name: 'train',
          signature: 'train(total_episodes, save_frequency=None, log_interval=100, **kwargs)',
          description: 'Train the multi-agent system',
          parameters: [
            { name: 'total_episodes', type: 'int', description: 'Total number of training episodes' },
            { name: 'save_frequency', type: 'int', description: 'Episodes between model saves (optional)' },
            { name: 'log_interval', type: 'int', description: 'Episodes between progress logs (default: 100)' },
            { name: 'tensorboard_logging', type: 'bool', description: 'Enable TensorBoard logging (default: False)' }
          ],
          returns: { 
            type: 'dict', 
            description: 'Training results including final_average_reward, total_training_time, convergence_episode' 
          },
          example: `# Basic training
results = controller.train(total_episodes=1000)

# Advanced training with logging
results = controller.train(
    total_episodes=5000,
    save_frequency=500,
    log_interval=50,
    tensorboard_logging=True
)

print(f"Final reward: {results['final_average_reward']:.3f}")
print(f"Training time: {results['total_training_time']:.1f}s")
print(f"Converged at episode: {results.get('convergence_episode', 'N/A')}")`
        },
        {
          name: 'evaluate',
          signature: 'evaluate(num_episodes=20, save_videos=False, deterministic=True, **kwargs)',
          description: 'Evaluate trained agents',
          parameters: [
            { name: 'num_episodes', type: 'int', description: 'Number of evaluation episodes (default: 20)' },
            { name: 'save_videos', type: 'bool', description: 'Save evaluation videos (default: False)' },
            { name: 'deterministic', type: 'bool', description: 'Use deterministic policy (default: True)' },
            { name: 'analyze_coordination', type: 'bool', description: 'Analyze agent coordination (default: False)' }
          ],
          returns: { 
            type: 'dict', 
            description: 'Evaluation results including average_reward, success_rate, coordination_score' 
          },
          example: `# Basic evaluation
eval_results = controller.evaluate(num_episodes=50)

# Comprehensive evaluation
eval_results = controller.evaluate(
    num_episodes=100,
    save_videos=True,
    analyze_coordination=True,
    detailed_analysis=True
)

print(f"Average reward: {eval_results['average_reward']:.3f}")
print(f"Success rate: {eval_results['success_rate']:.1%}")
print(f"Coordination score: {eval_results.get('coordination_score', 'N/A')}")`
        },
        {
          name: 'save_models',
          signature: 'save_models(save_path, include_replay_buffer=False)',
          description: 'Save trained models to disk',
          parameters: [
            { name: 'save_path', type: 'str', description: 'Path to save the models' },
            { name: 'include_replay_buffer', type: 'bool', description: 'Include replay buffer in save (default: False)' }
          ],
          returns: { type: 'str', description: 'Full path where models were saved' },
          example: `# Save models
model_path = controller.save_models("my_trained_agents")

# Save with replay buffer for resuming training
model_path = controller.save_models(
    "checkpoint_with_buffer",
    include_replay_buffer=True
)

print(f"Models saved to: {model_path}")`
        },
        {
          name: 'load_models',
          signature: 'load_models(load_path)',
          description: 'Load previously saved models',
          parameters: [
            { name: 'load_path', type: 'str', description: 'Path to the saved models' }
          ],
          returns: { type: 'None', description: 'Models loaded into controller' },
          example: `# Load pre-trained models
controller.load_models("my_trained_agents")

# Use loaded models for evaluation
eval_results = controller.evaluate(num_episodes=50)
print(f"Loaded model performance: {eval_results['average_reward']:.3f}")`
        }
      ]
    },
    {
      id: 'configuration',
      title: 'Configuration System',
      icon: <Settings className="w-5 h-5" />,
      description: 'Configuration management for algorithms and environments',
      methods: [
        {
          name: 'get_default_config',
          signature: 'get_default_config(algorithm)',
          description: 'Get default configuration for an algorithm',
          parameters: [
            { name: 'algorithm', type: 'str', description: 'Algorithm name (ippo, qmix, maddpg, etc.)' }
          ],
          returns: { type: 'dict', description: 'Default configuration dictionary' },
          example: `from easymarl.core.config_manager import get_default_config

# Get IPPO default config
ippo_config = get_default_config('ippo')
print(ippo_config)

# Customize and use
ippo_config['learning_rate'] = 1e-3
ippo_config['gamma'] = 0.99

controller = UnifiedMultiAgentController(
    env_name='MultiGrid-Empty-6x6',
    algorithm='ippo',
    config=ippo_config
)`
        },
        {
          name: 'validate_config',
          signature: 'validate_config(config, algorithm)',
          description: 'Validate configuration parameters',
          parameters: [
            { name: 'config', type: 'dict', description: 'Configuration to validate' },
            { name: 'algorithm', type: 'str', description: 'Target algorithm' }
          ],
          returns: { type: 'bool', description: 'True if configuration is valid' },
          example: `from easymarl.core.config_manager import validate_config

config = {
    'learning_rate': 3e-4,
    'batch_size': 256,
    'gamma': 0.99
}

is_valid = validate_config(config, 'ippo')
if is_valid:
    print("Configuration is valid!")
else:
    print("Configuration has errors")`
        }
      ]
    },
    {
      id: 'algorithms',
      title: 'Algorithm Interface',
      icon: <Zap className="w-5 h-5" />,
      description: 'Direct access to algorithm implementations',
      methods: [
        {
          name: 'select_actions',
          signature: 'select_actions(observations, deterministic=False)',
          description: 'Select actions for given observations',
          parameters: [
            { name: 'observations', type: 'list', description: 'List of agent observations' },
            { name: 'deterministic', type: 'bool', description: 'Use deterministic policy (default: False)' }
          ],
          returns: { type: 'list', description: 'List of selected actions for each agent' },
          example: `# Get actions from trained algorithm
observations = env.reset()
actions = controller.algorithm.select_actions(
    observations, 
    deterministic=True
)

# Execute actions
next_obs, rewards, done, info = env.step(actions)`
        },
        {
          name: 'update',
          signature: 'update(replay_buffer)',
          description: 'Update algorithm parameters',
          parameters: [
            { name: 'replay_buffer', type: 'ReplayBuffer', description: 'Experience replay buffer' }
          ],
          returns: { type: 'dict', description: 'Update statistics (loss, q_values, etc.)' },
          example: `# Manual training loop
for episode in range(1000):
    # Collect experience
    experience = collect_episode(env, controller.algorithm)
    replay_buffer.add(experience)
    
    # Update algorithm
    if len(replay_buffer) > batch_size:
        update_stats = controller.algorithm.update(replay_buffer)
        print(f"Loss: {update_stats.get('loss', 'N/A')}")`
        }
      ]
    },
    {
      id: 'environments',
      title: 'Environment Interface',
      icon: <Users className="w-5 h-5" />,
      description: 'Multi-agent environment management',
      methods: [
        {
          name: 'make_env',
          signature: 'make_env(env_name, n_envs=1, **kwargs)',
          description: 'Create vectorized multi-agent environment',
          parameters: [
            { name: 'env_name', type: 'str', description: 'Environment name' },
            { name: 'n_envs', type: 'int', description: 'Number of parallel environments' },
            { name: 'env_config', type: 'dict', description: 'Environment-specific configuration' }
          ],
          returns: { type: 'VectorizedMultiAgentEnv', description: 'Vectorized environment instance' },
          example: `from easymarl.environments import make_env

# Create single environment
env = make_env('MultiGrid-Empty-6x6', n_envs=1)

# Create vectorized environment
vectorized_env = make_env(
    'MultiGrid-Complex-12x12', 
    n_envs=8,
    env_config={
        'n_agents': 6,
        'max_steps': 200
    }
)

obs = vectorized_env.reset()
print(f"Observation shape: {obs[0].shape}")
print(f"Number of environments: {len(obs)}")`
        },
        {
          name: 'register_env',
          signature: 'register_env(name, env_class)',
          description: 'Register custom environment',
          parameters: [
            { name: 'name', type: 'str', description: 'Environment name identifier' },
            { name: 'env_class', type: 'class', description: 'Environment class to register' }
          ],
          returns: { type: 'None', description: 'Environment registered successfully' },
          example: `from easymarl.environments import register_env, MultiAgentEnv

class CustomEnv(MultiAgentEnv):
    def __init__(self, **kwargs):
        super().__init__()
        # Custom environment implementation
        
    def reset(self):
        # Return initial observations
        pass
    
    def step(self, actions):
        # Execute actions and return next state
        pass

# Register custom environment
register_env('Custom-v0', CustomEnv)

# Use with controller
controller = UnifiedMultiAgentController(
    env_name='Custom-v0',
    algorithm='ippo'
)`
        }
      ]
    },
    {
      id: 'utilities',
      title: 'Utility Functions',
      icon: <Code className="w-5 h-5" />,
      description: 'Helper functions and utilities',
      methods: [
        {
          name: 'get_available_algorithms',
          signature: 'get_available_algorithms()',
          description: 'Get list of available algorithms',
          parameters: [],
          returns: { type: 'list', description: 'List of available algorithm names' },
          example: `from easymarl.utils import get_available_algorithms

algorithms = get_available_algorithms()
print("Available algorithms:")
for algo in algorithms:
    print(f"  - {algo}")

# Use in dynamic selection
user_choice = input("Choose algorithm: ")
if user_choice in algorithms:
    controller = UnifiedMultiAgentController(
        env_name='MultiGrid-Empty-6x6',
        algorithm=user_choice
    )`
        },
        {
          name: 'get_available_environments',
          signature: 'get_available_environments()',
          description: 'Get list of available environments',
          parameters: [],
          returns: { type: 'list', description: 'List of available environment names' },
          example: `from easymarl.utils import get_available_environments

environments = get_available_environments()
print("Available environments:")
for env in environments:
    print(f"  - {env}")

# Filter by type
multigrid_envs = [env for env in environments if 'MultiGrid' in env]
print(f"MultiGrid environments: {len(multigrid_envs)}")`
        },
        {
          name: 'plot_training_curves',
          signature: 'plot_training_curves(training_data, save_path=None)',
          description: 'Plot training progress curves',
          parameters: [
            { name: 'training_data', type: 'dict', description: 'Training data with episode rewards' },
            { name: 'save_path', type: 'str', description: 'Path to save plot (optional)' }
          ],
          returns: { type: 'matplotlib.figure.Figure', description: 'Training curve plot' },
          example: `from easymarl.utils import plot_training_curves

# Train and collect data
results = controller.train(total_episodes=1000)

# Plot training curves
fig = plot_training_curves(
    results['training_data'],
    save_path='training_curves.png'
)

# Show plot
import matplotlib.pyplot as plt
plt.show()`
        }
      ]
    },
    {
      id: 'algorithm-factory',
      title: 'AlgorithmFactory',
      icon: <Settings className="w-5 h-5" />,
      description: 'Factory class for managing and creating MARL algorithms',
      methods: [
        {
          name: 'create_algorithm',
          signature: 'AlgorithmFactory.create_algorithm(algorithm_name, env, config, device)',
          description: 'Create an algorithm instance by name',
          parameters: [
            { name: 'algorithm_name', type: 'str', description: 'Name of the algorithm (qmix, ippo, maddpg, etc.)' },
            { name: 'env', type: 'gym.Env', description: 'Multi-agent environment instance' },
            { name: 'config', type: 'dict', description: 'Algorithm configuration parameters' },
            { name: 'device', type: 'torch.device', description: 'Device for computation (cpu/cuda)' }
          ],
          returns: { type: 'MARLAlgorithm', description: 'Algorithm instance ready for training' },
          example: `from easymarl.algorithms import AlgorithmFactory
from easymarl.environments import make_env
import torch

# Create environment
env = make_env('MultiGrid-Empty-8x8-v0', n_envs=4)

# Configuration
config = {
    'learning_rate': 3e-4,
    'batch_size': 32,
    'n_agents': 4
}

# Create algorithm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
algorithm = AlgorithmFactory.create_algorithm(
    algorithm_name='qmix',
    env=env,
    config=config,
    device=device
)

# Use algorithm
rollout_data = algorithm.collect_rollout(env)
metrics = algorithm.train_step(rollout_data)`
        },
        {
          name: 'get_available_algorithms',
          signature: 'AlgorithmFactory.get_available_algorithms()',
          description: 'Get list of all registered algorithms',
          parameters: [],
          returns: { type: 'list', description: 'List of algorithm names' },
          example: `from easymarl.algorithms import AlgorithmFactory

# Get all available algorithms
algorithms = AlgorithmFactory.get_available_algorithms()
print("Available algorithms:")
for algo in algorithms:
    print(f"  - {algo}")

# Check if specific algorithm is available
if 'qmix' in algorithms:
    print("QMIX is available!")

# Categorize algorithms
value_based = ['qmix', 'vdn', 'qtran', 'iql']
policy_based = ['ippo', 'mappo'] 
actor_critic = ['maddpg', 'coma']

print(f"Value-based: {[a for a in algorithms if a in value_based]}")
print(f"Policy-based: {[a for a in algorithms if a in policy_based]}")
print(f"Actor-critic: {[a for a in algorithms if a in actor_critic]}")`
        },
        {
          name: 'register_algorithm',
          signature: 'AlgorithmFactory.register_algorithm(name, algorithm_class)',
          description: 'Register a custom algorithm with the factory',
          parameters: [
            { name: 'name', type: 'str', description: 'Name for the algorithm' },
            { name: 'algorithm_class', type: 'class', description: 'Algorithm class inheriting from MARLAlgorithm' }
          ],
          returns: { type: 'None', description: 'No return value' },
          example: `from easymarl.algorithms import AlgorithmFactory
from easymarl.algorithms.base import MARLAlgorithm

class MyCustomAlgorithm(MARLAlgorithm):
    def __init__(self, env, config, device):
        super().__init__(env, config, device)
        # Your implementation here
        
    def _create_agents(self):
        # Create agents
        pass
        
    def collect_rollout(self, env):
        # Collect data
        pass
        
    def train_step(self, rollout_data):
        # Training step
        pass

# Register your algorithm
AlgorithmFactory.register_algorithm('my_custom', MyCustomAlgorithm)

# Now you can use it like any built-in algorithm
from easymarl.controllers import UnifiedMultiAgentController

controller = UnifiedMultiAgentController(
    env_name='MultiGrid-Empty-8x8-v0',
    algorithm='my_custom'  # Your custom algorithm
)`
        },
        {
          name: 'get_algorithm_info',
          signature: 'AlgorithmFactory.get_algorithm_info(algorithm_name)',
          description: 'Get detailed information about an algorithm',
          parameters: [
            { name: 'algorithm_name', type: 'str', description: 'Name of the algorithm' }
          ],
          returns: { type: 'dict', description: 'Algorithm information and capabilities' },
          example: `from easymarl.algorithms import AlgorithmFactory

# Get algorithm information
info = AlgorithmFactory.get_algorithm_info('qmix')
print(f"Algorithm: {info['name']}")
print(f"Type: {info['type']}")
print(f"Action Space: {info['action_space']}")
print(f"Best For: {info['best_for']}")
print(f"Pros: {info['pros']}")
print(f"Cons: {info['cons']}")

# Compare algorithms
algorithms = ['qmix', 'ippo', 'maddpg']
for algo in algorithms:
    info = AlgorithmFactory.get_algorithm_info(algo)
    print(f"{algo}: {info['description']}")`
        }
      ]
    },
    {
      id: 'configuration-system',
      title: 'Configuration System',
      icon: <Settings className="w-5 h-5" />,
      description: 'Advanced configuration management for algorithms and environments',
      methods: [
        {
          name: 'ConfigManager.load_config',
          signature: 'ConfigManager.load_config(algorithm_name, custom_params=None)',
          description: 'Load algorithm configuration with optional custom parameters',
          parameters: [
            { name: 'algorithm_name', type: 'str', description: 'Algorithm to configure' },
            { name: 'custom_params', type: 'dict', description: 'Custom parameter overrides' }
          ],
          returns: { type: 'dict', description: 'Complete configuration dictionary' },
          example: `from easymarl.core.config_manager import ConfigManager

# Load default QMIX configuration
config = ConfigManager.load_config('qmix')
print(f"Default learning rate: {config['learning_rate']}")

# Load with custom parameters
custom_config = ConfigManager.load_config(
    'qmix',
    custom_params={
        'learning_rate': 1e-3,
        'batch_size': 64,
        'mixer_hidden_dim': 256,
        'n_agents': 6
    }
)

# Use configuration
from easymarl.controllers import UnifiedMultiAgentController

controller = UnifiedMultiAgentController(
    env_name='MultiGrid-Custom-10x10',
    algorithm='qmix',
    config=custom_config
)`
        },
        {
          name: 'ConfigManager.save_config',
          signature: 'ConfigManager.save_config(config, filename)',
          description: 'Save configuration to YAML file for reuse',
          parameters: [
            { name: 'config', type: 'dict', description: 'Configuration to save' },
            { name: 'filename', type: 'str', description: 'Filename for the configuration' }
          ],
          returns: { type: 'str', description: 'Path where config was saved' },
          example: `from easymarl.core.config_manager import ConfigManager

# Create custom configuration
my_config = {
    'algorithm': 'qmix',
    'learning_rate': 5e-4,
    'batch_size': 128,
    'n_agents': 8,
    'cooperative': True,
    'centralized_training': True,
    'mixer_hidden_dim': 512,
    'network': {
        'hidden_dims': [256, 256],
        'activation': 'relu'
    },
    'training': {
        'max_episodes': 5000,
        'eval_frequency': 500
    }
}

# Save configuration
config_path = ConfigManager.save_config(
    my_config, 
    'my_qmix_setup.yaml'
)

# Load and use later
loaded_config = ConfigManager.load_config_file(config_path)
controller = UnifiedMultiAgentController(
    env_name='MultiGrid-Complex-12x12',
    algorithm='qmix',
    config=loaded_config
)`
        },
        {
          name: 'ConfigManager.get_template',
          signature: 'ConfigManager.get_template(algorithm_name)',
          description: 'Get the default parameter template for an algorithm',
          parameters: [
            { name: 'algorithm_name', type: 'str', description: 'Algorithm name' }
          ],
          returns: { type: 'dict', description: 'Default parameter template' },
          example: `from easymarl.core.config_manager import ConfigManager

# Get MAPPO template
template = ConfigManager.get_template('mappo')
print("MAPPO default parameters:")
for key, value in template['default_params'].items():
    print(f"  {key}: {value}")

# Modify template
template['default_params']['learning_rate'] = 1e-4
template['default_params']['ppo_epochs'] = 15
template['network_config']['hidden_dims'] = [512, 512]

# Use modified template
controller = UnifiedMultiAgentController(
    env_name='MultiGrid-Large-20x20',
    algorithm='mappo',
    config=template
)`
        },
        {
          name: 'ConfigManager.validate_config',
          signature: 'ConfigManager.validate_config(algorithm_name, config)',
          description: 'Validate configuration parameters for an algorithm',
          parameters: [
            { name: 'algorithm_name', type: 'str', description: 'Algorithm to validate for' },
            { name: 'config', type: 'dict', description: 'Configuration to validate' }
          ],
          returns: { type: 'tuple', description: '(is_valid, error_messages)' },
          example: `from easymarl.core.config_manager import ConfigManager

# Test configuration
test_config = {
    'learning_rate': 'invalid',  # Should be float
    'batch_size': -10,          # Should be positive
    'n_agents': 0               # Should be > 0
}

# Validate
is_valid, errors = ConfigManager.validate_config('qmix', test_config)

if not is_valid:
    print("Configuration errors:")
    for error in errors:
        print(f"  - {error}")
else:
    print("Configuration is valid!")

# Fix configuration
fixed_config = {
    'learning_rate': 3e-4,
    'batch_size': 32, 
    'n_agents': 4
}

is_valid, errors = ConfigManager.validate_config('qmix', fixed_config)
print(f"Fixed config valid: {is_valid}")`
        }
      ]
    },
    {
      id: 'base-classes',
      title: 'Base Classes for Custom Algorithms',
      icon: <Code className="w-5 h-5" />,
      description: 'Abstract base classes for implementing custom MARL algorithms',
      methods: [
        {
          name: 'MARLAgent.__init__',
          signature: 'MARLAgent.__init__(agent_id, obs_space, action_space, config)',
          description: 'Initialize a multi-agent reinforcement learning agent',
          parameters: [
            { name: 'agent_id', type: 'int', description: 'Unique identifier for the agent' },
            { name: 'obs_space', type: 'gym.Space', description: 'Observation space definition' },
            { name: 'action_space', type: 'gym.Space', description: 'Action space definition' },
            { name: 'config', type: 'dict', description: 'Agent configuration parameters' }
          ],
          returns: { type: 'None', description: 'No return value' },
          example: `from easymarl.algorithms.base import MARLAgent
import torch.nn as nn

class MyAgent(MARLAgent):
    def __init__(self, agent_id, obs_space, action_space, config):
        super().__init__(agent_id, obs_space, action_space, config)
        
        # Your custom initialization
        self.hidden_dim = config.get('hidden_dim', 128)
        self.learning_rate = config.get('learning_rate', 1e-3)
        
        # Define neural networks
        self.policy_net = nn.Sequential(
            nn.Linear(obs_space.shape[0], self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, action_space.n)
        )
        
        # Define optimizer
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(),
            lr=self.learning_rate
        )`
        },
        {
          name: 'MARLAgent.get_action',
          signature: 'get_action(observation, training=True)',
          description: 'Select action given observation (must implement)',
          parameters: [
            { name: 'observation', type: 'dict', description: 'Agent observation' },
            { name: 'training', type: 'bool', description: 'Whether in training mode' }
          ],
          returns: { type: 'tuple', description: '(action, additional_info)' },
          example: `def get_action(self, observation, training=True):
    """Select action using policy network"""
    import torch
    import torch.nn.functional as F
    
    with torch.no_grad():
        obs_tensor = torch.FloatTensor(observation['vector'])
        action_logits = self.policy_net(obs_tensor)
        
        if training:
            # Sample from policy
            action_probs = F.softmax(action_logits, dim=-1)
            action = torch.multinomial(action_probs, 1).item()
            log_prob = torch.log(action_probs[action])
        else:
            # Greedy action
            action = torch.argmax(action_logits).item()
            log_prob = 0.0
            
    return action, log_prob`
        },
        {
          name: 'MARLAgent.update',
          signature: 'update(batch_data)',
          description: 'Learn from batch of experiences (must implement)',
          parameters: [
            { name: 'batch_data', type: 'dict', description: 'Batch of experience data' }
          ],
          returns: { type: 'dict', description: 'Training metrics' },
          example: `def update(self, batch_data):
    """Update agent using batch data"""
    import torch
    import torch.nn.functional as F
    
    # Extract data
    observations = torch.FloatTensor(batch_data['observations'])
    actions = torch.LongTensor(batch_data['actions'])
    rewards = torch.FloatTensor(batch_data['rewards'])
    
    # Forward pass
    action_logits = self.policy_net(observations)
    log_probs = F.log_softmax(action_logits, dim=-1)
    selected_log_probs = log_probs.gather(1, actions.unsqueeze(1))
    
    # Compute loss (simple REINFORCE)
    loss = -(selected_log_probs.squeeze() * rewards).mean()
    
    # Backward pass
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
    
    return {
        'loss': loss.item(),
        'mean_reward': rewards.mean().item()
    }`
        },
        {
          name: 'MARLAlgorithm.__init__',
          signature: 'MARLAlgorithm.__init__(env, config, device)',
          description: 'Initialize multi-agent algorithm coordinator',
          parameters: [
            { name: 'env', type: 'gym.Env', description: 'Multi-agent environment' },
            { name: 'config', type: 'dict', description: 'Algorithm configuration' },
            { name: 'device', type: 'torch.device', description: 'Computation device' }
          ],
          returns: { type: 'None', description: 'No return value' },
          example: `from easymarl.algorithms.base import MARLAlgorithm

class MyAlgorithm(MARLAlgorithm):
    def __init__(self, env, config, device):
        super().__init__(env, config, device)
        
        # Algorithm-specific initialization
        self.n_agents = env.n_agents
        self.obs_space = env.observation_space
        self.action_space = env.action_space
        self.config = config
        self.device = device
        
        # Create agents
        self._create_agents()
        
        # Algorithm state
        self.step_count = 0
        self.total_episodes = 0`
        },
        {
          name: 'MARLAlgorithm.collect_rollout',
          signature: 'collect_rollout(env)',
          description: 'Collect experience data from environment (must implement)',
          parameters: [
            { name: 'env', type: 'gym.Env', description: 'Environment to collect from' }
          ],
          returns: { type: 'dict', description: 'Collected experience data' },
          example: `def collect_rollout(self, env):
    """Collect experience rollout"""
    rollout_data = {
        'observations': [],
        'actions': [],
        'rewards': [],
        'dones': []
    }
    
    obs = env.reset()
    for step in range(self.rollout_length):
        # Get actions from all agents
        actions = []
        for i, agent in enumerate(self.agents):
            action, _ = agent.get_action(obs[i])
            actions.append(action)
        
        # Environment step
        next_obs, rewards, dones, infos = env.step(actions)
        
        # Store data
        rollout_data['observations'].append(obs)
        rollout_data['actions'].append(actions)
        rollout_data['rewards'].append(rewards)
        rollout_data['dones'].append(dones)
        
        obs = next_obs
        if any(dones):
            break
            
    return rollout_data`
        },
        {
          name: 'MARLAlgorithm.train_step',
          signature: 'train_step(rollout_data)',
          description: 'Perform training step with collected data (must implement)',
          parameters: [
            { name: 'rollout_data', type: 'dict', description: 'Data from collect_rollout' }
          ],
          returns: { type: 'dict', description: 'Training metrics' },
          example: `def train_step(self, rollout_data):
    """Training step using rollout data"""
    metrics = {}
    
    # Process data for each agent
    for i, agent in enumerate(self.agents):
        # Extract agent-specific data
        agent_data = {
            'observations': [obs[i]['vector'] for obs in rollout_data['observations']],
            'actions': [actions[i] for actions in rollout_data['actions']],
            'rewards': [rewards[i] for rewards in rollout_data['rewards']]
        }
        
        # Update agent
        agent_metrics = agent.update(agent_data)
        metrics[f'agent_{i}'] = agent_metrics
    
    self.step_count += 1
    return metrics`
        }
      ]
    },
    {
      id: 'research-interface',
      title: 'Research Interface',
      icon: <Brain className="w-5 h-5" />,
      description: 'Advanced research tools for algorithm discovery, experimentation, and comparison',
      methods: [
        {
          name: 'ResearchInterface.create_experiment',
          signature: 'create_experiment(name, description="", algorithm="ippo", environment="MultiGrid-Cluttered-Fixed-15x15")',
          description: 'Create a structured research experiment with comprehensive configuration',
          parameters: [
            { name: 'name', type: 'str', description: 'Experiment name for identification' },
            { name: 'description', type: 'str', description: 'Detailed experiment description' },
            { name: 'algorithm', type: 'str', description: 'MARL algorithm to test' },
            { name: 'environment', type: 'str', description: 'Environment for experimentation' }
          ],
          returns: { type: 'ExperimentConfig', description: 'Configured experiment ready for execution' },
          example: `from easymarl.core.research_interface import ResearchInterface

# Create research interface
research = ResearchInterface()

# Create comprehensive experiment
experiment = research.create_experiment(
    name="cooperative_coordination_study",
    description="Comparing value decomposition methods for cooperative multi-agent tasks",
    algorithm="qmix",
    environment="MultiGrid-Cooperative-12x12"
)

# Configure experiment parameters
experiment.hyperparameters = {
    'learning_rate': 5e-4,
    'batch_size': 64,
    'mixer_hidden_dim': 256,
    'epsilon_decay': 0.9995
}

experiment.training_config = {
    'max_episodes': 5000,
    'evaluation_frequency': 500,
    'save_frequency': 1000
}

# Advanced research options
experiment.research_options = {
    'track_agent_states': True,
    'record_action_distributions': True,
    'analyze_coordination_patterns': True,
    'compare_with_baselines': ['vdn', 'ippo']
}`
        },
        {
          name: 'AlgorithmDiscovery.search_algorithms',
          signature: 'search_algorithms(category=None, type=None, action_space=None, complexity=None)',
          description: 'Intelligent algorithm search and discovery based on research criteria',
          parameters: [
            { name: 'category', type: 'str', description: 'Algorithm category (Value-Based, Policy-Based, Actor-Critic)' },
            { name: 'type', type: 'str', description: 'Scenario type (Cooperative, Competitive, Mixed)' },
            { name: 'action_space', type: 'str', description: 'Action space type (Discrete, Continuous)' },
            { name: 'complexity', type: 'str', description: 'Implementation complexity (Beginner, Intermediate, Advanced, Expert)' }
          ],
          returns: { type: 'list', description: 'List of matching algorithms' },
          example: `from easymarl.core.research_interface import AlgorithmDiscovery

discovery = AlgorithmDiscovery()

# Search for cooperative value-based algorithms
cooperative_algos = discovery.search_algorithms(
    category='Value-Based',
    type='Cooperative',
    action_space='Discrete'
)
print(f"Cooperative value-based algorithms: {cooperative_algos}")

# Search for beginner-friendly algorithms
beginner_algos = discovery.search_algorithms(complexity='Beginner')
print(f"Beginner algorithms: {beginner_algos}")

# Search for continuous action algorithms
continuous_algos = discovery.search_algorithms(action_space='Continuous')
print(f"Continuous action algorithms: {continuous_algos}")

# Get detailed information about found algorithms
for algo in cooperative_algos:
    info = discovery.get_algorithm_info(algo)
    print(f"\\n{algo.upper()}:")
    print(f"  Paper: {info['paper']}")
    print(f"  Strengths: {info['strengths']}")
    print(f"  Best for: {info['research_applications']}")`
        },
        {
          name: 'AlgorithmDiscovery.get_research_recommendations',
          signature: 'get_research_recommendations(research_focus, experience_level="intermediate")',
          description: 'Get algorithm recommendations tailored to specific research objectives',
          parameters: [
            { name: 'research_focus', type: 'str', description: 'Research area (cooperative_ai, competitive_ai, continuous_control, sample_efficiency, scalability)' },
            { name: 'experience_level', type: 'str', description: 'Researcher experience level (beginner, intermediate, advanced, expert)' }
          ],
          returns: { type: 'dict', description: 'Recommended and alternative algorithms' },
          example: `# Get research recommendations
cooperative_recommendations = discovery.get_research_recommendations(
    research_focus='cooperative_ai',
    experience_level='intermediate'
)

print("Recommended algorithms:", cooperative_recommendations['recommended'])
print("Alternative algorithms:", cooperative_recommendations['alternatives'])

# Sample efficiency focused research
efficiency_recommendations = discovery.get_research_recommendations(
    research_focus='sample_efficiency',
    experience_level='advanced'
)

# Competitive AI research recommendations
competitive_recommendations = discovery.get_research_recommendations(
    research_focus='competitive_ai',
    experience_level='beginner'
)

# Plan research progression
research_plan = []
for level in ['beginner', 'intermediate', 'advanced', 'expert']:
    recs = discovery.get_research_recommendations('cooperative_ai', level)
    research_plan.append({
        'level': level,
        'algorithms': recs['recommended']
    })

print("Research progression plan:")
for stage in research_plan:
    print(f"  {stage['level']}: {stage['algorithms']}")`
        },
        {
          name: 'AlgorithmDiscovery.compare_algorithms',
          signature: 'compare_algorithms(algorithms)',
          description: 'Comprehensive algorithm comparison across multiple research dimensions',
          parameters: [
            { name: 'algorithms', type: 'list', description: 'List of algorithm names to compare' }
          ],
          returns: { type: 'dict', description: 'Detailed comparison matrix' },
          example: `# Compare cooperative algorithms
comparison = discovery.compare_algorithms(['qmix', 'vdn', 'mappo', 'coma'])

print("Algorithm Comparison:")
print("=" * 80)
print(f"{'Algorithm':<10} {'Category':<20} {'Complexity':<12} {'Sample Eff.':<12} {'Scalability':<12}")
print("=" * 80)

for algo, details in comparison.items():
    print(f"{algo.upper():<10} "
          f"{details['category']:<20} "
          f"{details['complexity']:<12} "
          f"{details['sample_efficiency']:<12} "
          f"{details['scalability']:<12}")

# Detailed feature comparison
print("\\nDetailed Strengths and Limitations:")
for algo, details in comparison.items():
    print(f"\\n{algo.upper()}:")
    print(f"  Strengths: {', '.join(details['strengths'])}")
    print(f"  Limitations: {', '.join(details['limitations'])}")
    print(f"  Key Hyperparameters: {', '.join(details['key_hyperparameters'])}")

# Find best algorithm for specific criteria
best_for_efficiency = max(comparison.keys(), 
                         key=lambda x: comparison[x]['sample_efficiency'])
print(f"\\nBest for sample efficiency: {best_for_efficiency}")`
        },
        {
          name: 'ResearchInterface.get_parameter_schema',
          signature: 'get_parameter_schema(algorithm)',
          description: 'Get comprehensive parameter schema for GUI generation and validation',
          parameters: [
            { name: 'algorithm', type: 'str', description: 'Algorithm name' }
          ],
          returns: { type: 'dict', description: 'Complete parameter schema with validation rules' },
          example: `# Get parameter schema for dynamic GUI generation
schema = research.get_parameter_schema('qmix')

print("QMIX Parameter Schema:")
print(json.dumps(schema, indent=2))

# Extract parameter ranges for optimization
param_ranges = {}
for category, params in schema.items():
    if isinstance(params, dict):
        for param_name, param_info in params.items():
            if isinstance(param_info, dict) and 'min' in param_info:
                param_ranges[param_name] = {
                    'min': param_info['min'],
                    'max': param_info['max'],
                    'default': param_info['default'],
                    'type': param_info['type']
                }

print("\\nParameter ranges for optimization:")
for param, range_info in param_ranges.items():
    print(f"  {param}: {range_info}")

# Generate random configurations for hyperparameter search
import random

def generate_random_config(schema):
    config = {}
    for category, params in schema.items():
        if isinstance(params, dict):
            for param_name, param_info in params.items():
                if isinstance(param_info, dict) and 'min' in param_info:
                    if param_info['type'] == 'float':
                        value = random.uniform(param_info['min'], param_info['max'])
                    elif param_info['type'] == 'int':
                        value = random.randint(param_info['min'], param_info['max'])
                    config[param_name] = value
    return config

# Generate 5 random configurations
random_configs = [generate_random_config(schema) for _ in range(5)]
print(f"\\nGenerated {len(random_configs)} random configurations for testing")`
        },
        {
          name: 'HyperparameterOptimizer.optimize',
          signature: 'optimize(algorithm, environment, parameter_ranges, optimization_method="random_search")',
          description: 'Advanced hyperparameter optimization for MARL algorithms',
          parameters: [
            { name: 'algorithm', type: 'str', description: 'Algorithm to optimize' },
            { name: 'environment', type: 'str', description: 'Environment for evaluation' },
            { name: 'parameter_ranges', type: 'dict', description: 'Parameter search ranges' },
            { name: 'optimization_method', type: 'str', description: 'Optimization strategy (random_search, grid_search, bayesian)' }
          ],
          returns: { type: 'dict', description: 'Optimization results with best parameters' },
          example: `from easymarl.core.research_interface import HyperparameterOptimizer

optimizer = HyperparameterOptimizer()

# Define parameter search space
parameter_ranges = {
    'learning_rate': {'min': 1e-5, 'max': 1e-2, 'type': 'log_uniform'},
    'batch_size': {'values': [16, 32, 64, 128], 'type': 'categorical'},
    'mixer_hidden_dim': {'min': 64, 'max': 512, 'step': 64, 'type': 'int'},
    'epsilon_decay': {'min': 0.99, 'max': 0.9999, 'type': 'uniform'},
    'gamma': {'min': 0.95, 'max': 0.99, 'type': 'uniform'}
}

# Run hyperparameter optimization
optimization_results = optimizer.optimize(
    algorithm='qmix',
    environment='MultiGrid-Cooperative-8x8',
    parameter_ranges=parameter_ranges,
    optimization_method='bayesian',
    n_trials=50,
    evaluation_episodes=100
)

print("Optimization Results:")
print(f"Best parameters: {optimization_results['best_params']}")
print(f"Best performance: {optimization_results['best_score']}")
print(f"Optimization time: {optimization_results['optimization_time']}")

# Analyze optimization trajectory
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(optimization_results['scores_history'])
plt.title('Optimization Progress')
plt.xlabel('Trial')
plt.ylabel('Performance Score')

plt.subplot(1, 2, 2)
plt.plot(optimization_results['best_scores_history'])
plt.title('Best Score Over Time')
plt.xlabel('Trial')
plt.ylabel('Best Score So Far')

plt.tight_layout()
plt.show()

# Use optimized parameters
from easymarl.controllers import UnifiedMultiAgentController

optimized_controller = UnifiedMultiAgentController(
    env_name='MultiGrid-Cooperative-8x8',
    algorithm='qmix',
    config=optimization_results['best_params'],
    n_envs=8
)

final_results = optimized_controller.train(total_episodes=3000)
print(f"Final optimized performance: {final_results['final_reward']}")`
        }
      ]
    }
  ];

  const filteredSections = apiSections.map(section => ({
    ...section,
    methods: section.methods.filter(method =>
      method.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      method.description.toLowerCase().includes(searchTerm.toLowerCase())
    )
  })).filter(section => section.methods.length > 0 || searchTerm === '');

  return (
    <div className="min-h-screen pt-24 pb-20 bg-gray-50">
      <div className="max-w-7xl mx-auto px-4">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-5xl md:text-6xl font-bold mb-6">
            <span className="gradient-text">API Reference</span>
          </h1>
          <p className="text-xl text-gray-600 max-w-4xl mx-auto mb-8">
            Complete API documentation for EasyMARL. Find all classes, methods, and parameters 
            you need to build powerful multi-agent systems.
          </p>
        </div>

        {/* Search */}
        <div className="max-w-2xl mx-auto mb-12">
          <div className="relative">
            <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
            <input
              type="text"
              placeholder="Search API methods, classes, or descriptions..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-12 pr-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
          </div>
        </div>

        {/* API Sections */}
        <div className="space-y-6">
          {filteredSections.map((section) => (
            <div key={section.id} className="bg-white rounded-lg shadow-lg overflow-hidden">
              {/* Section Header */}
              <div
                className="flex items-center justify-between p-6 bg-gradient-to-r from-blue-50 to-purple-50 border-b cursor-pointer hover:from-blue-100 hover:to-purple-100 transition-colors"
                onClick={() => toggleSection(section.id)}
              >
                <div className="flex items-center space-x-3">
                  <div className="p-2 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg text-white">
                    {section.icon}
                  </div>
                  <div>
                    <h2 className="text-2xl font-bold">{section.title}</h2>
                    <p className="text-gray-600">{section.description}</p>
                  </div>
                </div>
                {expandedSections[section.id] ? (
                  <ChevronDown className="w-6 h-6 text-gray-500" />
                ) : (
                  <ChevronRight className="w-6 h-6 text-gray-500" />
                )}
              </div>

              {/* Section Content */}
              {expandedSections[section.id] && (
                <div className="p-6">
                  <div className="grid gap-6">
                    {section.methods.map((method, idx) => (
                      <div
                        key={idx}
                        className="border border-gray-200 rounded-lg p-6 hover:shadow-md transition-shadow cursor-pointer"
                        onClick={() => setSelectedMethod(method)}
                      >
                        {/* Method Header */}
                        <div className="flex items-start justify-between mb-4">
                          <div>
                            <h3 className="text-xl font-bold text-blue-600 mb-2">
                              {method.name}
                            </h3>
                            <code className="text-sm bg-gray-100 px-3 py-1 rounded text-gray-800">
                              {method.signature}
                            </code>
                          </div>
                          <ArrowRight className="w-5 h-5 text-gray-400 mt-1" />
                        </div>

                        <p className="text-gray-700 mb-4">{method.description}</p>

                        {/* Quick Parameter Preview */}
                        <div className="flex flex-wrap gap-2">
                          {method.parameters.slice(0, 3).map((param) => (
                            <span
                              key={param.name}
                              className="px-2 py-1 bg-blue-100 text-blue-700 rounded-md text-xs"
                            >
                              {param.name}: {param.type}
                            </span>
                          ))}
                          {method.parameters.length > 3 && (
                            <span className="px-2 py-1 bg-gray-100 text-gray-600 rounded-md text-xs">
                              +{method.parameters.length - 3} more
                            </span>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>

        {/* Method Detail Modal */}
        {selectedMethod && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
            <div className="bg-white rounded-lg shadow-2xl p-8 max-w-4xl w-full max-h-[90vh] overflow-y-auto">
              {/* Modal Header */}
              <div className="flex items-center justify-between mb-6">
                <div>
                  <h2 className="text-3xl font-bold text-blue-600 mb-2">
                    {selectedMethod.name}
                  </h2>
                  <code className="text-sm bg-gray-100 px-3 py-2 rounded text-gray-800">
                    {selectedMethod.signature}
                  </code>
                </div>
                <button
                  onClick={() => setSelectedMethod(null)}
                  className="text-gray-500 hover:text-gray-700 text-xl"
                >
                  ✕
                </button>
              </div>

              <p className="text-gray-700 mb-6">{selectedMethod.description}</p>

              {/* Parameters */}
              <div className="mb-6">
                <h3 className="text-xl font-semibold mb-4 flex items-center">
                  <Settings className="w-5 h-5 mr-2 text-blue-600" />
                  Parameters
                </h3>
                <div className="space-y-3">
                  {selectedMethod.parameters.map((param, idx) => (
                    <div key={idx} className="flex items-start space-x-4 p-3 bg-gray-50 rounded-lg">
                      <div className="flex-shrink-0">
                        <span className="px-2 py-1 bg-blue-100 text-blue-700 rounded text-sm font-mono">
                          {param.name}
                        </span>
                      </div>
                      <div className="flex-grow">
                        <div className="flex items-center space-x-2 mb-1">
                          <Type className="w-4 h-4 text-gray-500" />
                          <span className="text-sm font-medium text-gray-600">{param.type}</span>
                        </div>
                        <p className="text-gray-700 text-sm">{param.description}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Returns */}
              <div className="mb-6">
                <h3 className="text-xl font-semibold mb-4 flex items-center">
                  <ArrowRight className="w-5 h-5 mr-2 text-green-600" />
                  Returns
                </h3>
                <div className="p-3 bg-green-50 rounded-lg">
                  <div className="flex items-center space-x-2 mb-2">
                    <Type className="w-4 h-4 text-green-600" />
                    <span className="font-medium text-green-700">{selectedMethod.returns.type}</span>
                  </div>
                  <p className="text-gray-700">{selectedMethod.returns.description}</p>
                </div>
              </div>

              {/* Example */}
              <div className="mb-6">
                <h3 className="text-xl font-semibold mb-4 flex items-center">
                  <Code className="w-5 h-5 mr-2 text-purple-600" />
                  Example Usage
                </h3>
                <CodeBlock code={selectedMethod.example} language="python" />
              </div>

              {/* Close Button */}
              <div className="flex justify-end">
                <button
                  onClick={() => setSelectedMethod(null)}
                  className="px-6 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors"
                >
                  Close
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Quick Start Guide */}
        <div className="mt-12 bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg p-8">
          <h2 className="text-3xl font-bold mb-6 flex items-center">
            <Play className="w-8 h-8 mr-3 text-blue-600" />
            Quick Start Guide
          </h2>
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <h3 className="text-xl font-semibold mb-3">Basic Usage Pattern</h3>
              <CodeBlock
                code={`from easymarl.controllers import UnifiedMultiAgentController

# 1. Create controller
controller = UnifiedMultiAgentController(
    env_name='MultiGrid-Empty-6x6',
    algorithm='ippo',
    n_envs=8
)

# 2. Train agents
results = controller.train(total_episodes=1000)

# 3. Evaluate performance
eval_results = controller.evaluate(num_episodes=20)

# 4. Save trained models
controller.save_models("my_agents")`}
                language="python"
              />
            </div>
            <div>
              <h3 className="text-xl font-semibold mb-3">Key Features</h3>
              <ul className="space-y-3">
                <li className="flex items-center space-x-2">
                  <CheckCircle className="w-5 h-5 text-green-500" />
                  <span>Unified API for all algorithms</span>
                </li>
                <li className="flex items-center space-x-2">
                  <CheckCircle className="w-5 h-5 text-green-500" />
                  <span>Automatic vectorization support</span>
                </li>
                <li className="flex items-center space-x-2">
                  <CheckCircle className="w-5 h-5 text-green-500" />
                  <span>Educational mode for learning</span>
                </li>
                <li className="flex items-center space-x-2">
                  <CheckCircle className="w-5 h-5 text-green-500" />
                  <span>Built-in performance monitoring</span>
                </li>
                <li className="flex items-center space-x-2">
                  <CheckCircle className="w-5 h-5 text-green-500" />
                  <span>Easy model saving/loading</span>
                </li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default APIReferencePage;
