import React, { useState } from 'react';
import { 
  Book, 
  Search, 
  ChevronDown, 
  ChevronRight, 
  Code, 
  Settings, 
  Zap, 
  Users,
  Play,
  FileText,
  Hash,
  Type,
  ArrowRight,
  Info,
  AlertCircle,
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
