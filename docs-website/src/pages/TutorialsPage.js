import React, { useState } from 'react';
import { 
  Play, 
  Clock, 
  User, 
  Users, 
  Zap, 
  BookOpen, 
  CheckCircle, 
  ArrowRight,
  Code,
  Monitor,
  Settings,
  TrendingUp,
  Brain,
  Target
} from 'lucide-react';
import CodeBlock from '../components/CodeBlock';

const TutorialsPage = () => {
  const [selectedTutorial, setSelectedTutorial] = useState(null);

  const tutorials = [
    {
      id: 'quickstart',
      title: 'Quick Start Guide',
      level: 'Beginner',
      duration: '10 minutes',
      icon: <Play className="w-6 h-6" />,
      description: 'Get up and running with EasyMARL in minutes. Your first multi-agent training.',
      objectives: [
        'Install EasyMARL',
        'Run your first MARL algorithm',
        'Understand the basic workflow',
        'Visualize training results'
      ],
      content: {
        steps: [
          {
            title: 'Installation',
            code: `# Install EasyMARL
pip install easymarl

# Or install from source
pip install git+https://github.com/shreyanmitra/EasyMARL.git`,
            explanation: 'Install EasyMARL using pip. This includes all core algorithms and environments.'
          },
          {
            title: 'First Training Session',
            code: `from easymarl.controllers import UnifiedMultiAgentController

# Create controller with educational mode
controller = UnifiedMultiAgentController(
    env_name='MultiGrid-Empty-6x6',
    algorithm='ippo',
    n_envs=1,  # Single environment for learning
    educational_mode=True  # Detailed explanations
)

# Train agents
results = controller.train(total_episodes=100)
print(f"Training complete! Final reward: {results['final_average_reward']:.3f}")`,
            explanation: 'This creates a simple multi-agent environment with 4 IPPO agents. Educational mode provides detailed explanations of what\'s happening.'
          },
          {
            title: 'Evaluate Performance',
            code: `# Evaluate trained agents
eval_results = controller.evaluate(num_episodes=10, save_videos=True)
print(f"Evaluation reward: {eval_results['average_reward']:.3f}")
print(f"Success rate: {eval_results['success_rate']:.1%}")`,
            explanation: 'Test your trained agents and optionally save videos to see how they behave.'
          }
        ]
      }
    },
    {
      id: 'educational-mode',
      title: 'Educational Mode Deep Dive',
      level: 'Beginner',
      duration: '20 minutes',
      icon: <BookOpen className="w-6 h-6" />,
      description: 'Learn MARL concepts through EasyMARL\'s educational features.',
      objectives: [
        'Understand educational mode features',
        'Learn MARL concepts step-by-step',
        'Interpret training metrics',
        'Debug common issues'
      ],
      content: {
        steps: [
          {
            title: 'Enabling Educational Mode',
            code: `controller = UnifiedMultiAgentController(
    env_name='MultiGrid-Empty-8x8',
    algorithm='ippo',
    educational_mode=True,  # Enable learning features
    n_envs=1,              # Single env for clarity
    config={
        'max_steps': 500,     # Longer episodes
        'log_interval': 1     # Log every episode
    }
)`,
            explanation: 'Educational mode provides step-by-step explanations, longer episodes for observation, and frequent logging.'
          },
          {
            title: 'Understanding Training Output',
            code: `# Educational mode will print:
# 🎓 Episode 1 Walkthrough:
#    - Agents will observe the environment state
#    - Each agent will choose actions based on current policy
#    - Environment will update and provide rewards
#    - Algorithm will learn from this experience

results = controller.train(total_episodes=50)`,
            explanation: 'Educational mode explains each step of the learning process, helping you understand what agents are doing.'
          },
          {
            title: 'Interpreting Metrics',
            code: `# Understanding the output:
# 📊 Episode 10/50 Summary:
#    Recent Avg Reward: 2.340
#    Recent Avg Length: 87.5 steps
#    Episode Duration: 0.45s
#    Training Speed: 133.3 episodes/min

# 🎓 Understanding the Metrics:
#    - Reward shows how well agents are performing
#    - Length shows how long episodes last
#    - Improving performance trend`,
            explanation: 'Learn to read and interpret the training metrics to understand agent performance and learning progress.'
          }
        ]
      }
    },
    {
      id: 'vectorized-training',
      title: 'High-Performance Vectorized Training',
      level: 'Intermediate',
      duration: '25 minutes',
      icon: <Zap className="w-6 h-6" />,
      description: 'Scale up your training with vectorized environments for 8x speedup.',
      objectives: [
        'Understand vectorization benefits',
        'Configure parallel environments',
        'Optimize training performance',
        'Monitor system resources'
      ],
      content: {
        steps: [
          {
            title: 'Vectorized Setup',
            code: `# High-performance training setup
controller = UnifiedMultiAgentController(
    env_name='MultiGrid-Complex-12x12',
    algorithm='qmix',
    n_envs=8,  # 8 parallel environments
    enable_performance_monitoring=True,
    config={
        'batch_size': 64,
        'update_frequency': 4
    }
)`,
            explanation: 'Vectorized training runs multiple environments in parallel, dramatically reducing training time.'
          },
          {
            title: 'Performance Comparison',
            code: `# Single environment training
single_controller = UnifiedMultiAgentController(
    env_name='MultiGrid-Complex-12x12',
    algorithm='qmix',
    n_envs=1
)

# Vectorized training (8x faster data collection)
vectorized_controller = UnifiedMultiAgentController(
    env_name='MultiGrid-Complex-12x12',
    algorithm='qmix',
    n_envs=8  # 8x speedup
)

# Train both and compare times
import time
start = time.time()
vectorized_results = vectorized_controller.train(total_episodes=1000)
vectorized_time = time.time() - start

print(f"Vectorized training completed in {vectorized_time:.1f}s")`,
            explanation: 'Vectorized training provides significant speedup by collecting data from multiple environments simultaneously.'
          },
          {
            title: 'Advanced Configuration',
            code: `# Advanced vectorized setup
controller = UnifiedMultiAgentController(
    env_name='MultiGrid-Advanced-15x15',
    algorithm='maddpg',
    n_envs=16,  # Scale up further
    enable_advanced_tracking=True,
    enable_performance_monitoring=True,
    config={
        'normalize_observations': True,
        'domain_randomization': True,
        'batch_size': 128
    }
)`,
            explanation: 'Advanced configurations can include observation normalization, domain randomization, and larger batch sizes for better performance.'
          }
        ]
      }
    },
    {
      id: 'algorithm-selection',
      title: 'Choosing the Right Algorithm',
      level: 'Intermediate',
      duration: '30 minutes',
      icon: <Brain className="w-6 h-6" />,
      description: 'Learn when and why to use different MARL algorithms.',
      objectives: [
        'Understand algorithm categories',
        'Match algorithms to problems',
        'Compare performance characteristics',
        'Handle different action spaces'
      ],
      content: {
        steps: [
          {
            title: 'Algorithm Categories',
            code: `# Policy-Based: Good for continuous actions and exploration
ippo_controller = UnifiedMultiAgentController(
    env_name='Continuous-Control-Env',
    algorithm='ippo',  # Independent learning
    n_envs=8
)

# Value-Based: Good for discrete actions and coordination
qmix_controller = UnifiedMultiAgentController(
    env_name='MultiGrid-Cooperative-8x8',
    algorithm='qmix',  # Value factorization
    n_envs=8
)

# Actor-Critic: Good for mixed scenarios
maddpg_controller = UnifiedMultiAgentController(
    env_name='Mixed-Action-Env',
    algorithm='maddpg',  # Centralized training
    n_envs=4
)`,
            explanation: 'Different algorithm types excel in different scenarios. Choose based on your action space and coordination requirements.'
          },
          {
            title: 'Problem-Algorithm Matching',
            code: `# Simple coordination tasks
simple_controller = UnifiedMultiAgentController(
    env_name='MultiGrid-Simple-6x6',
    algorithm='vdn',  # Simple value decomposition
    educational_mode=True
)

# Complex coordination with credit assignment
complex_controller = UnifiedMultiAgentController(
    env_name='MultiGrid-Complex-12x12',
    algorithm='coma',  # Counterfactual reasoning
    n_envs=4
)

# Large-scale systems
scale_controller = UnifiedMultiAgentController(
    env_name='LargeScale-Environment',
    algorithm='mfpo',  # Mean field approximation
    n_envs=8
)`,
            explanation: 'Match algorithm complexity to problem complexity. Start simple and scale up as needed.'
          },
          {
            title: 'Performance Comparison',
            code: `# Compare multiple algorithms
algorithms = ['ippo', 'qmix', 'maddpg']
results = {}

for algo in algorithms:
    controller = UnifiedMultiAgentController(
        env_name='MultiGrid-Benchmark-8x8',
        algorithm=algo,
        n_envs=8
    )
    
    result = controller.train(total_episodes=1000)
    eval_result = controller.evaluate(num_episodes=20)
    
    results[algo] = {
        'final_reward': eval_result['average_reward'],
        'training_time': result['total_training_time'],
        'convergence_episode': result.get('convergence_episode', 1000)
    }

# Print comparison
for algo, metrics in results.items():
    print(f"{algo}: Reward={metrics['final_reward']:.3f}, "
          f"Time={metrics['training_time']:.1f}s")`,
            explanation: 'Empirically compare algorithms on your specific problem to find the best fit.'
          }
        ]
      }
    },
    {
      id: 'custom-environments',
      title: 'Creating Custom Environments',
      level: 'Advanced',
      duration: '45 minutes',
      icon: <Settings className="w-6 h-6" />,
      description: 'Build custom multi-agent environments for your specific use case.',
      objectives: [
        'Understand environment structure',
        'Implement custom observations',
        'Design reward functions',
        'Integrate with EasyMARL'
      ],
      content: {
        steps: [
          {
            title: 'Basic Environment Structure',
            code: `import gym
import numpy as np
from easymarl.environments import MultiAgentEnv

class CustomMultiAgentEnv(MultiAgentEnv):
    def __init__(self, n_agents=4, grid_size=8):
        super().__init__()
        self.n_agents = n_agents
        self.grid_size = grid_size
        
        # Define observation and action spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=1, 
            shape=(grid_size, grid_size, 3), 
            dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(5)  # 4 directions + stay
        
    def reset(self):
        # Initialize environment state
        self.state = np.zeros((self.grid_size, self.grid_size, 3))
        # Return initial observations for all agents
        return self.get_observations()
    
    def step(self, actions):
        # Execute actions and update environment
        rewards = self.compute_rewards(actions)
        observations = self.get_observations()
        done = self.is_done()
        info = self.get_info()
        
        return observations, rewards, done, info`,
            explanation: 'Custom environments inherit from MultiAgentEnv and implement the standard gym interface with multi-agent extensions.'
          },
          {
            title: 'Reward Function Design',
            code: `def compute_rewards(self, actions):
    """Design rewards to encourage desired behaviors"""
    rewards = np.zeros(self.n_agents)
    
    for i, action in enumerate(actions):
        # Individual task completion reward
        if self.agent_reached_goal(i):
            rewards[i] += 10.0
        
        # Cooperation bonus
        nearby_agents = self.count_nearby_agents(i)
        rewards[i] += nearby_agents * 0.5
        
        # Penalty for collisions
        if self.agent_collision(i):
            rewards[i] -= 2.0
            
        # Small step penalty to encourage efficiency
        rewards[i] -= 0.01
    
    # Team reward component
    team_bonus = self.compute_team_performance()
    rewards += team_bonus
    
    return rewards`,
            explanation: 'Design reward functions that balance individual and team objectives. Include shaping rewards to guide learning.'
          },
          {
            title: 'Integration with EasyMARL',
            code: `# Register your custom environment
from easymarl.environments import register_env

register_env('CustomMultiAgent-v0', CustomMultiAgentEnv)

# Use with UnifiedMultiAgentController
controller = UnifiedMultiAgentController(
    env_name='CustomMultiAgent-v0',
    algorithm='qmix',
    n_envs=8,
    config={
        'env_config': {
            'n_agents': 6,
            'grid_size': 10
        }
    }
)

# Train on your custom environment
results = controller.train(total_episodes=2000)`,
            explanation: 'Register your environment with EasyMARL to use it seamlessly with all algorithms and features.'
          }
        ]
      }
    },
    {
      id: 'production-deployment',
      title: 'Production Deployment',
      level: 'Advanced',
      duration: '40 minutes',
      icon: <Monitor className="w-6 h-6" />,
      description: 'Deploy trained models to production with monitoring and scaling.',
      objectives: [
        'Export trained models',
        'Set up inference pipeline',
        'Monitor model performance',
        'Handle model updates'
      ],
      content: {
        steps: [
          {
            title: 'Model Export and Saving',
            code: `# Train and save model
controller = UnifiedMultiAgentController(
    env_name='MultiGrid-Production-10x10',
    algorithm='qmix',
    n_envs=16,
    production_mode=True  # Optimized for production
)

results = controller.train(total_episodes=5000)

# Save production model
model_path = controller.save_models("production_model_v1")
print(f"Model saved to: {model_path}")

# Save deployment config
import json
deployment_config = {
    'algorithm': 'qmix',
    'env_name': 'MultiGrid-Production-10x10',
    'model_path': model_path,
    'n_agents': controller.n_agents,
    'observation_space': str(controller.observation_space),
    'action_space': str(controller.action_space)
}

with open('deployment_config.json', 'w') as f:
    json.dump(deployment_config, f, indent=2)`,
            explanation: 'Save trained models with complete configuration for reproducible deployment.'
          },
          {
            title: 'Inference Pipeline',
            code: `class ProductionInference:
    def __init__(self, config_path, model_path):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Load trained model
        self.controller = UnifiedMultiAgentController(
            env_name=self.config['env_name'],
            algorithm=self.config['algorithm'],
            n_envs=1,  # Single environment for inference
            training=False  # Inference mode
        )
        self.controller.load_models(model_path)
        
    def predict(self, observations):
        """Get actions for given observations"""
        with torch.no_grad():
            actions = self.controller.algorithm.select_actions(
                observations, 
                deterministic=True  # Deterministic for production
            )
        return actions
    
    def predict_batch(self, observation_batch):
        """Handle batch predictions efficiently"""
        batch_actions = []
        for obs in observation_batch:
            actions = self.predict(obs)
            batch_actions.append(actions)
        return batch_actions

# Usage
inference = ProductionInference('deployment_config.json', model_path)
actions = inference.predict(current_observations)`,
            explanation: 'Create a production inference pipeline that loads models and handles predictions efficiently.'
          },
          {
            title: 'Monitoring and Performance',
            code: `import time
import logging
from collections import deque

class ProductionMonitor:
    def __init__(self, window_size=1000):
        self.metrics = {
            'prediction_times': deque(maxlen=window_size),
            'prediction_count': 0,
            'error_count': 0
        }
        
    def log_prediction(self, prediction_time, success=True):
        self.metrics['prediction_times'].append(prediction_time)
        self.metrics['prediction_count'] += 1
        
        if not success:
            self.metrics['error_count'] += 1
            
        # Log performance every 100 predictions
        if self.metrics['prediction_count'] % 100 == 0:
            self.report_performance()
    
    def report_performance(self):
        avg_time = np.mean(self.metrics['prediction_times'])
        error_rate = self.metrics['error_count'] / self.metrics['prediction_count']
        
        logging.info(f"Performance Report:")
        logging.info(f"  Average prediction time: {avg_time:.3f}s")
        logging.info(f"  Error rate: {error_rate:.3%}")
        logging.info(f"  Predictions per second: {1/avg_time:.1f}")

# Integration with inference
monitor = ProductionMonitor()

def monitored_predict(observations):
    start_time = time.time()
    try:
        actions = inference.predict(observations)
        prediction_time = time.time() - start_time
        monitor.log_prediction(prediction_time, success=True)
        return actions
    except Exception as e:
        prediction_time = time.time() - start_time
        monitor.log_prediction(prediction_time, success=False)
        logging.error(f"Prediction error: {e}")
        raise`,
            explanation: 'Monitor production performance including prediction times, error rates, and throughput metrics.'
          }
        ]
      }
    }
  ];

  const getDifficultyColor = (level) => {
    switch(level) {
      case 'Beginner': return 'bg-green-100 text-green-700 border-green-200';
      case 'Intermediate': return 'bg-yellow-100 text-yellow-700 border-yellow-200';
      case 'Advanced': return 'bg-red-100 text-red-700 border-red-200';
      default: return 'bg-gray-100 text-gray-700 border-gray-200';
    }
  };

  return (
    <div className="min-h-screen pt-24 pb-20 bg-gray-50">
      <div className="max-w-7xl mx-auto px-4">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-5xl md:text-6xl font-bold mb-6">
            <span className="gradient-text">Tutorials</span>
          </h1>
          <p className="text-xl text-gray-600 max-w-4xl mx-auto mb-8">
            Step-by-step tutorials to master Multi-Agent Reinforcement Learning with EasyMARL.
            From basic concepts to advanced deployment strategies.
          </p>
        </div>

        {/* Tutorial Selection */}
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6 mb-12">
          {tutorials.map((tutorial) => (
            <div
              key={tutorial.id}
              className={`bg-white rounded-lg shadow-lg p-6 cursor-pointer transition-all duration-300 hover:shadow-xl border-2 ${
                selectedTutorial?.id === tutorial.id ? 'border-blue-500' : 'border-transparent'
              }`}
              onClick={() => setSelectedTutorial(tutorial)}
            >
              <div className="flex items-center space-x-3 mb-4">
                <div className="p-2 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg text-white">
                  {tutorial.icon}
                </div>
                <div>
                  <h3 className="text-lg font-bold">{tutorial.title}</h3>
                  <div className="flex items-center space-x-2 text-sm text-gray-600">
                    <Clock className="w-4 h-4" />
                    <span>{tutorial.duration}</span>
                  </div>
                </div>
              </div>
              
              <p className="text-gray-700 mb-4">{tutorial.description}</p>
              
              <div className="flex items-center justify-between">
                <span className={`px-3 py-1 rounded-full text-xs font-medium border ${getDifficultyColor(tutorial.level)}`}>
                  {tutorial.level}
                </span>
                <button className="text-blue-600 hover:text-blue-800 font-medium text-sm flex items-center space-x-1">
                  <span>Start Tutorial</span>
                  <ArrowRight className="w-4 h-4" />
                </button>
              </div>
            </div>
          ))}
        </div>

        {/* Tutorial Content */}
        {selectedTutorial && (
          <div className="bg-white rounded-lg shadow-lg p-8">
            <div className="flex items-center space-x-4 mb-6">
              <div className="p-3 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg text-white">
                {selectedTutorial.icon}
              </div>
              <div>
                <h2 className="text-3xl font-bold">{selectedTutorial.title}</h2>
                <div className="flex items-center space-x-4 text-gray-600">
                  <div className="flex items-center space-x-1">
                    <Clock className="w-4 h-4" />
                    <span>{selectedTutorial.duration}</span>
                  </div>
                  <span className={`px-3 py-1 rounded-full text-xs font-medium border ${getDifficultyColor(selectedTutorial.level)}`}>
                    {selectedTutorial.level}
                  </span>
                </div>
              </div>
            </div>

            {/* Learning Objectives */}
            <div className="mb-8 p-6 bg-blue-50 rounded-lg">
              <h3 className="text-xl font-semibold mb-4 flex items-center">
                <Target className="w-5 h-5 mr-2 text-blue-600" />
                Learning Objectives
              </h3>
              <ul className="space-y-2">
                {selectedTutorial.objectives.map((objective, idx) => (
                  <li key={idx} className="flex items-center space-x-2">
                    <CheckCircle className="w-5 h-5 text-green-500" />
                    <span>{objective}</span>
                  </li>
                ))}
              </ul>
            </div>

            {/* Tutorial Steps */}
            <div className="space-y-8">
              {selectedTutorial.content.steps.map((step, idx) => (
                <div key={idx} className="border-l-4 border-blue-500 pl-6">
                  <h3 className="text-xl font-semibold mb-3 flex items-center">
                    <div className="w-8 h-8 bg-blue-500 text-white rounded-full flex items-center justify-center text-sm mr-3">
                      {idx + 1}
                    </div>
                    {step.title}
                  </h3>
                  <p className="text-gray-700 mb-4">{step.explanation}</p>
                  <CodeBlock code={step.code} language="python" />
                </div>
              ))}
            </div>

            {/* Next Steps */}
            <div className="mt-12 p-6 bg-gradient-to-r from-green-50 to-blue-50 rounded-lg">
              <h3 className="text-xl font-semibold mb-3 flex items-center">
                <TrendingUp className="w-5 h-5 mr-2 text-green-600" />
                Next Steps
              </h3>
              <p className="text-gray-700 mb-4">
                Congratulations on completing this tutorial! Here are some recommended next steps:
              </p>
              <ul className="space-y-2">
                <li className="flex items-center space-x-2">
                  <ArrowRight className="w-4 h-4 text-green-600" />
                  <span>Try the code examples in your own environment</span>
                </li>
                <li className="flex items-center space-x-2">
                  <ArrowRight className="w-4 h-4 text-green-600" />
                  <span>Explore the API reference for advanced features</span>
                </li>
                <li className="flex items-center space-x-2">
                  <ArrowRight className="w-4 h-4 text-green-600" />
                  <span>Check out more advanced tutorials</span>
                </li>
                <li className="flex items-center space-x-2">
                  <ArrowRight className="w-4 h-4 text-green-600" />
                  <span>Join the community for questions and discussions</span>
                </li>
              </ul>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default TutorialsPage;
