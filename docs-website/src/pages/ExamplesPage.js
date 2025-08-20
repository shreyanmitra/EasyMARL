import React, { useState } from 'react';
import { 
  PlayCircle, 
  Download, 
  Eye, 
  Star, 
  Clock, 
  Settings,
  Brain,
  Target,
  BarChart3,
  Code
} from 'lucide-react';
import CodeBlock from '../components/CodeBlock';

const ExamplesPage = () => {
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [selectedExample, setSelectedExample] = useState(null);

  const categories = [
    { id: 'all', label: 'All Examples', icon: <Code className="w-4 h-4" /> },
    { id: 'quickstart', label: 'Quick Start', icon: <PlayCircle className="w-4 h-4" /> },
    { id: 'algorithms', label: 'Algorithms', icon: <Brain className="w-4 h-4" /> },
    { id: 'environments', label: 'Environments', icon: <Target className="w-4 h-4" /> },
    { id: 'advanced', label: 'Advanced', icon: <Settings className="w-4 h-4" /> }
  ];

  const examples = [
    {
      id: 'basic-ippo',
      category: 'quickstart',
      title: 'Basic IPPO Training',
      description: 'Simple independent PPO training on MultiGrid environment',
      difficulty: 'Beginner',
      time: '5 min',
      featured: true,
      tags: ['IPPO', 'Policy-Based', 'MultiGrid'],
      code: `from easymarl.controllers import UnifiedMultiAgentController

# Create controller for basic IPPO training
controller = UnifiedMultiAgentController(
    env_name='MultiGrid-Empty-6x6',
    algorithm='ippo',
    n_envs=1,
    educational_mode=True,
    config={
        'learning_rate': 3e-4,
        'max_grad_norm': 0.5,
        'gamma': 0.99,
        'gae_lambda': 0.95
    }
)

# Train agents
print("🚀 Starting IPPO training...")
results = controller.train(total_episodes=200)

print(f"✅ Training complete!")
print(f"📊 Final average reward: {results['final_average_reward']:.3f}")
print(f"⏱️ Training time: {results['total_training_time']:.1f}s")

# Evaluate trained agents
eval_results = controller.evaluate(num_episodes=10, save_videos=True)
print(f"🎯 Evaluation reward: {eval_results['average_reward']:.3f}")`,
      explanation: 'This example shows the simplest way to train agents using Independent PPO. Perfect for beginners getting started with MARL.',
      useCase: 'Use when you want agents to learn independently without explicit coordination.'
    },
    {
      id: 'qmix-coordination',
      category: 'algorithms',
      title: 'QMIX Coordination',
      description: 'Value factorization for coordinated multi-agent learning',
      difficulty: 'Intermediate',
      time: '10 min',
      featured: true,
      tags: ['QMIX', 'Value-Based', 'Coordination'],
      code: `from easymarl.controllers import UnifiedMultiAgentController

# QMIX for coordination tasks
controller = UnifiedMultiAgentController(
    env_name='MultiGrid-Cooperative-8x8',
    algorithm='qmix',
    n_envs=8,  # Vectorized for speed
    config={
        'mixer_hidden_dim': 64,
        'exploration_epsilon_start': 1.0,
        'exploration_epsilon_end': 0.05,
        'exploration_decay_episodes': 1000,
        'target_update_frequency': 200
    }
)

print("🤝 Starting QMIX coordination training...")
results = controller.train(total_episodes=2000)

# Analyze coordination
eval_results = controller.evaluate(
    num_episodes=20, 
    save_videos=True,
    analyze_coordination=True
)

print(f"🎯 Team Performance:")
print(f"   Success Rate: {eval_results['success_rate']:.1%}")
print(f"   Coordination Score: {eval_results['coordination_score']:.3f}")
print(f"   Average Episode Length: {eval_results['avg_episode_length']:.1f}")`,
      explanation: 'QMIX learns to coordinate agents by factorizing team value functions. Ideal for tasks requiring tight coordination.',
      useCase: 'Use for cooperative tasks where agents need to coordinate actions for optimal team performance.'
    },
    {
      id: 'maddpg-continuous',
      category: 'algorithms',
      title: 'MADDPG Continuous Control',
      description: 'Multi-agent actor-critic for continuous action spaces',
      difficulty: 'Advanced',
      time: '15 min',
      featured: false,
      tags: ['MADDPG', 'Actor-Critic', 'Continuous'],
      code: `from easymarl.controllers import UnifiedMultiAgentController

# MADDPG for continuous control
controller = UnifiedMultiAgentController(
    env_name='ContinuousMultiAgent-v0',
    algorithm='maddpg',
    n_envs=4,
    config={
        'actor_lr': 1e-3,
        'critic_lr': 1e-3,
        'tau': 0.01,  # Soft update rate
        'batch_size': 256,
        'replay_buffer_size': 1000000,
        'noise_std': 0.1
    }
)

print("🎮 Starting MADDPG continuous control training...")

# Train with experience replay
results = controller.train(
    total_episodes=3000,
    save_frequency=500,
    log_tensorboard=True
)

# Test in deterministic mode
controller.algorithm.set_noise_level(0.0)  # No exploration noise
eval_results = controller.evaluate(
    num_episodes=50,
    deterministic=True
)

print(f"🎯 Deterministic Performance: {eval_results['average_reward']:.3f}")`,
      explanation: 'MADDPG uses centralized training with decentralized execution for continuous action spaces.',
      useCase: 'Perfect for robotics, autonomous vehicles, and other continuous control multi-agent tasks.'
    },
    {
      id: 'vectorized-training',
      category: 'advanced',
      title: 'High-Speed Vectorized Training',
      description: '16x parallel environments for maximum training speed',
      difficulty: 'Intermediate',
      time: '12 min',
      featured: true,
      tags: ['Vectorization', 'Performance', 'Scaling'],
      code: `from easymarl.controllers import UnifiedMultiAgentController
import time

# High-performance vectorized setup
controller = UnifiedMultiAgentController(
    env_name='MultiGrid-Complex-12x12',
    algorithm='mappo',  # Multi-Agent PPO
    n_envs=16,  # 16 parallel environments!
    enable_performance_monitoring=True,
    config={
        'batch_size': 512,
        'mini_batch_size': 128,
        'update_epochs': 4,
        'clip_range': 0.2
    }
)

print("⚡ Starting high-speed vectorized training...")
start_time = time.time()

# Monitor training performance
results = controller.train(
    total_episodes=5000,
    performance_logging=True
)

training_time = time.time() - start_time
episodes_per_second = 5000 / training_time

print(f"🚀 Performance Metrics:")
print(f"   Total Training Time: {training_time:.1f}s")
print(f"   Episodes/Second: {episodes_per_second:.1f}")
print(f"   Sample Efficiency: {results['sample_efficiency']:.3f}")
print(f"   Final Reward: {results['final_average_reward']:.3f}")

# Performance analysis
perf_stats = controller.get_performance_stats()
print(f"   Avg Step Time: {perf_stats['avg_step_time']:.4f}s")
print(f"   Memory Usage: {perf_stats['memory_usage_mb']:.1f} MB")`,
      explanation: 'Vectorized training dramatically speeds up learning by running multiple environments in parallel.',
      useCase: 'Use when you need fast training and have sufficient computational resources.'
    },
    {
      id: 'custom-environment',
      category: 'environments',
      title: 'Custom Environment Integration',
      description: 'Create and integrate your own multi-agent environment',
      difficulty: 'Advanced',
      time: '20 min',
      featured: false,
      tags: ['Custom Environment', 'Integration', 'Advanced'],
      code: `import gym
import numpy as np
from easymarl.environments import MultiAgentEnv, register_env

class ResourceCollectionEnv(MultiAgentEnv):
    """Custom environment: agents collect resources collaboratively"""
    
    def __init__(self, n_agents=4, grid_size=10, n_resources=20):
        super().__init__()
        self.n_agents = n_agents
        self.grid_size = grid_size
        self.n_resources = n_resources
        
        # Define spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=1, 
            shape=(grid_size, grid_size, 4),  # agents, resources, walls, visited
            dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(5)  # 4 directions + collect
        
        self.reset()
    
    def reset(self):
        """Reset environment to initial state"""
        self.agent_positions = np.random.randint(0, self.grid_size, (self.n_agents, 2))
        self.resource_positions = np.random.randint(0, self.grid_size, (self.n_resources, 2))
        self.collected_resources = 0
        self.step_count = 0
        return self.get_observations()
    
    def step(self, actions):
        """Execute one step with given actions"""
        self.step_count += 1
        rewards = np.zeros(self.n_agents)
        
        for i, action in enumerate(actions):
            old_pos = self.agent_positions[i].copy()
            
            # Move agent
            if action < 4:  # Movement actions
                self.move_agent(i, action)
            elif action == 4:  # Collect action
                collected = self.collect_resource(i)
                if collected:
                    rewards[i] += 5.0  # Individual reward
                    rewards += 1.0     # Team bonus
        
        # Check if episode is done
        done = (self.collected_resources >= self.n_resources) or (self.step_count >= 200)
        
        # Small step penalty to encourage efficiency
        rewards -= 0.01
        
        observations = self.get_observations()
        info = {'collected_resources': self.collected_resources}
        
        return observations, rewards, done, info
    
    def get_observations(self):
        """Generate observations for all agents"""
        obs = []
        for i in range(self.n_agents):
            agent_obs = np.zeros((self.grid_size, self.grid_size, 4))
            
            # Agent positions (channel 0)
            for j, pos in enumerate(self.agent_positions):
                agent_obs[pos[0], pos[1], 0] = 1.0 if j == i else 0.5
            
            # Resource positions (channel 1)
            for pos in self.resource_positions:
                if tuple(pos) not in self.collected_positions:
                    agent_obs[pos[0], pos[1], 1] = 1.0
            
            obs.append(agent_obs)
        
        return obs

# Register the custom environment
register_env('ResourceCollection-v0', ResourceCollectionEnv)

# Use with EasyMARL
controller = UnifiedMultiAgentController(
    env_name='ResourceCollection-v0',
    algorithm='qmix',
    n_envs=8,
    config={
        'env_config': {
            'n_agents': 6,
            'grid_size': 12,
            'n_resources': 30
        }
    }
)

print("🌍 Training on custom Resource Collection environment...")
results = controller.train(total_episodes=3000)

print(f"✅ Custom environment training complete!")
print(f"📊 Final team performance: {results['final_average_reward']:.3f}")`,
      explanation: 'This example shows how to create a custom multi-agent environment and integrate it with EasyMARL.',
      useCase: 'Use when you need specific environment dynamics not available in standard environments.'
    },
    {
      id: 'educational-deep-dive',
      category: 'quickstart',
      title: 'Educational Mode Deep Dive',
      description: 'Learn MARL concepts with detailed explanations',
      difficulty: 'Beginner',
      time: '8 min',
      featured: false,
      tags: ['Educational', 'Learning', 'Concepts'],
      code: `from easymarl.controllers import UnifiedMultiAgentController

# Enable educational mode for learning
controller = UnifiedMultiAgentController(
    env_name='MultiGrid-Simple-6x6',
    algorithm='ippo',
    n_envs=1,  # Single env for clarity
    educational_mode=True,  # Detailed explanations
    config={
        'log_interval': 5,      # Log every 5 episodes
        'save_frequency': 50,   # Save every 50 episodes
        'max_steps': 200       # Longer episodes for observation
    }
)

print("🎓 Starting educational training session...")
print("    Educational mode will explain each step!")

# Training with detailed explanations
results = controller.train(total_episodes=100)

print("\\n📚 What you learned:")
print("   - How agents observe their environment")
print("   - How policies are updated based on experience")
print("   - How rewards shape agent behavior")
print("   - How independent learning works in MARL")

# Detailed evaluation with analysis
eval_results = controller.evaluate(
    num_episodes=10,
    save_videos=True,
    detailed_analysis=True
)

print("\\n🔍 Detailed Analysis:")
print(f"   Agent Coordination: {eval_results.get('coordination_metric', 'N/A')}")
print(f"   Learning Efficiency: {eval_results.get('learning_efficiency', 'N/A')}")
print(f"   Exploration Balance: {eval_results.get('exploration_metric', 'N/A')}")`,
      explanation: 'Educational mode provides step-by-step explanations of MARL concepts during training.',
      useCase: 'Perfect for students and researchers new to multi-agent reinforcement learning.'
    },
    {
      id: 'algorithm-comparison',
      category: 'algorithms',
      title: 'Algorithm Performance Comparison',
      description: 'Compare multiple algorithms on the same task',
      difficulty: 'Intermediate',
      time: '25 min',
      featured: true,
      tags: ['Comparison', 'Benchmarking', 'Analysis'],
      code: `from easymarl.controllers import UnifiedMultiAgentController
import matplotlib.pyplot as plt
import pandas as pd

# Algorithms to compare
algorithms = ['ippo', 'qmix', 'maddpg', 'mappo']
results_comparison = {}

print("🔬 Starting algorithm comparison study...")

for algo in algorithms:
    print(f"\\n🧪 Testing {algo.upper()}...")
    
    controller = UnifiedMultiAgentController(
        env_name='MultiGrid-Cooperative-8x8',
        algorithm=algo,
        n_envs=8,
        config={
            'total_episodes': 2000,
            'log_interval': 100,
            'evaluation_frequency': 200
        }
    )
    
    # Train and collect metrics
    results = controller.train(total_episodes=2000)
    eval_results = controller.evaluate(num_episodes=20)
    
    results_comparison[algo] = {
        'final_reward': eval_results['average_reward'],
        'training_time': results['total_training_time'],
        'sample_efficiency': results.get('sample_efficiency', 0),
        'convergence_episode': results.get('convergence_episode', 2000),
        'success_rate': eval_results.get('success_rate', 0)
    }

# Display comparison results
print("\\n📊 Algorithm Comparison Results:")
print("=" * 80)
print(f"{'Algorithm':<10} {'Reward':<8} {'Time(s)':<8} {'Efficiency':<10} {'Success%':<8}")
print("=" * 80)

for algo, metrics in results_comparison.items():
    print(f"{algo.upper():<10} "
          f"{metrics['final_reward']:<8.3f} "
          f"{metrics['training_time']:<8.1f} "
          f"{metrics['sample_efficiency']:<10.3f} "
          f"{metrics['success_rate']*100:<8.1f}")

# Find best algorithm
best_algo = max(results_comparison.keys(), 
               key=lambda x: results_comparison[x]['final_reward'])
print(f"\\n🏆 Best performing algorithm: {best_algo.upper()}")

# Save results for further analysis
df = pd.DataFrame(results_comparison).T
df.to_csv('algorithm_comparison.csv')
print("💾 Results saved to 'algorithm_comparison.csv'")`,
      explanation: 'Systematically compare different algorithms to find the best one for your specific task.',
      useCase: 'Use when you need to choose the optimal algorithm for a specific multi-agent problem.'
    },
    {
      id: 'production-deployment',
      category: 'advanced',
      title: 'Production Model Deployment',
      description: 'Deploy trained models for real-world applications',
      difficulty: 'Advanced',
      time: '30 min',
      featured: false,
      tags: ['Production', 'Deployment', 'Inference'],
      code: `from easymarl.controllers import UnifiedMultiAgentController
import torch
import numpy as np
import time
import json

# Train production-ready model
controller = UnifiedMultiAgentController(
    env_name='MultiGrid-Production-10x10',
    algorithm='qmix',
    n_envs=16,
    production_mode=True,  # Optimized for production
    config={
        'batch_size': 256,
        'replay_buffer_size': 100000,
        'target_update_frequency': 500
    }
)

print("🏭 Training production model...")
results = controller.train(total_episodes=10000)

# Save model for deployment
model_path = controller.save_models("production_qmix_v1")
print(f"💾 Model saved to: {model_path}")

# Create deployment configuration
deployment_config = {
    'model_info': {
        'algorithm': 'qmix',
        'version': 'v1.0',
        'training_episodes': 10000,
        'performance': results['final_average_reward']
    },
    'model_path': model_path,
    'environment': 'MultiGrid-Production-10x10',
    'n_agents': 4,
    'action_space_size': 5
}

with open('deployment_config.json', 'w') as f:
    json.dump(deployment_config, f, indent=2)

# Production inference class
class ProductionMARL:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Load trained model for inference
        self.controller = UnifiedMultiAgentController(
            env_name=self.config['environment'],
            algorithm=self.config['model_info']['algorithm'],
            n_envs=1,
            training=False  # Inference mode
        )
        self.controller.load_models(self.config['model_path'])
        
        # Performance monitoring
        self.inference_times = []
        self.prediction_count = 0
        
    def predict(self, observations, deterministic=True):
        """Get actions for given observations"""
        start_time = time.time()
        
        with torch.no_grad():
            actions = self.controller.algorithm.select_actions(
                observations, 
                deterministic=deterministic
            )
        
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        self.prediction_count += 1
        
        return actions
    
    def get_performance_stats(self):
        """Get inference performance statistics"""
        if not self.inference_times:
            return None
            
        return {
            'avg_inference_time': np.mean(self.inference_times),
            'predictions_per_second': 1.0 / np.mean(self.inference_times),
            'total_predictions': self.prediction_count,
            'min_inference_time': np.min(self.inference_times),
            'max_inference_time': np.max(self.inference_times)
        }

# Initialize production system
production_system = ProductionMARL('deployment_config.json')

# Simulate production usage
print("\\n🚀 Simulating production inference...")
for i in range(100):
    # Simulate incoming observations
    dummy_observations = [np.random.rand(10, 10, 3) for _ in range(4)]
    actions = production_system.predict(dummy_observations)
    
    if i % 20 == 0:
        print(f"   Processed {i+1} inference requests...")

# Performance report
stats = production_system.get_performance_stats()
print(f"\\n📈 Production Performance Report:")
print(f"   Average inference time: {stats['avg_inference_time']*1000:.2f}ms")
print(f"   Predictions per second: {stats['predictions_per_second']:.1f}")
print(f"   Total predictions: {stats['total_predictions']}")
print(f"   Latency range: {stats['min_inference_time']*1000:.2f}-{stats['max_inference_time']*1000:.2f}ms")`,
      explanation: 'Complete pipeline for deploying trained MARL models in production environments.',
      useCase: 'Use when deploying trained agents in real-world applications requiring low latency and high reliability.'
    }
  ];

  const getDifficultyColor = (difficulty) => {
    switch(difficulty) {
      case 'Beginner': return 'bg-green-100 text-green-700 border-green-200';
      case 'Intermediate': return 'bg-yellow-100 text-yellow-700 border-yellow-200';
      case 'Advanced': return 'bg-red-100 text-red-700 border-red-200';
      default: return 'bg-gray-100 text-gray-700 border-gray-200';
    }
  };

  const filteredExamples = selectedCategory === 'all' 
    ? examples 
    : examples.filter(example => example.category === selectedCategory);

  return (
    <div className="min-h-screen pt-24 pb-20 bg-gray-50">
      <div className="max-w-7xl mx-auto px-4">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-5xl md:text-6xl font-bold mb-6">
            <span className="gradient-text">Code Examples</span>
          </h1>
          <p className="text-xl text-gray-600 max-w-4xl mx-auto mb-8">
            Ready-to-run code examples for every MARL use case. From simple quickstarts to advanced 
            production deployments, find the perfect starting point for your project.
          </p>
        </div>

        {/* Category Filter */}
        <div className="flex flex-wrap justify-center gap-3 mb-12">
          {categories.map((category) => (
            <button
              key={category.id}
              onClick={() => setSelectedCategory(category.id)}
              className={`flex items-center space-x-2 px-4 py-2 rounded-lg font-medium transition-all duration-200 ${
                selectedCategory === category.id
                  ? 'bg-blue-600 text-white shadow-lg'
                  : 'bg-white text-gray-700 hover:bg-gray-100 border border-gray-200'
              }`}
            >
              {category.icon}
              <span>{category.label}</span>
            </button>
          ))}
        </div>

        {/* Featured Examples */}
        {selectedCategory === 'all' && (
          <div className="mb-12">
            <h2 className="text-3xl font-bold mb-6 flex items-center">
              <Star className="w-8 h-8 mr-3 text-yellow-500" />
              Featured Examples
            </h2>
            <div className="grid md:grid-cols-2 gap-6">
              {examples.filter(ex => ex.featured).map((example) => (
                <div
                  key={example.id}
                  className="bg-white rounded-lg shadow-lg p-6 border-l-4 border-yellow-500 hover:shadow-xl transition-all duration-300 cursor-pointer"
                  onClick={() => setSelectedExample(example)}
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-bold">{example.title}</h3>
                    <Star className="w-5 h-5 text-yellow-500" />
                  </div>
                  <p className="text-gray-600 mb-4">{example.description}</p>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-4 text-sm text-gray-500">
                      <div className="flex items-center space-x-1">
                        <Clock className="w-4 h-4" />
                        <span>{example.time}</span>
                      </div>
                      <span className={`px-2 py-1 rounded-full text-xs font-medium border ${getDifficultyColor(example.difficulty)}`}>
                        {example.difficulty}
                      </span>
                    </div>
                    <button className="text-blue-600 hover:text-blue-800 font-medium text-sm">
                      View Example →
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* All Examples Grid */}
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6 mb-12">
          {filteredExamples.map((example) => (
            <div
              key={example.id}
              className="bg-white rounded-lg shadow-lg p-6 hover:shadow-xl transition-all duration-300 cursor-pointer"
              onClick={() => setSelectedExample(example)}
            >
              <div className="flex items-start justify-between mb-3">
                <h3 className="text-lg font-bold">{example.title}</h3>
                {example.featured && <Star className="w-5 h-5 text-yellow-500" />}
              </div>
              
              <p className="text-gray-600 mb-4 text-sm">{example.description}</p>
              
              <div className="flex flex-wrap gap-2 mb-4">
                {example.tags.slice(0, 2).map((tag) => (
                  <span key={tag} className="px-2 py-1 bg-blue-100 text-blue-700 rounded-md text-xs">
                    {tag}
                  </span>
                ))}
                {example.tags.length > 2 && (
                  <span className="px-2 py-1 bg-gray-100 text-gray-600 rounded-md text-xs">
                    +{example.tags.length - 2} more
                  </span>
                )}
              </div>
              
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3 text-sm text-gray-500">
                  <div className="flex items-center space-x-1">
                    <Clock className="w-4 h-4" />
                    <span>{example.time}</span>
                  </div>
                </div>
                <span className={`px-3 py-1 rounded-full text-xs font-medium border ${getDifficultyColor(example.difficulty)}`}>
                  {example.difficulty}
                </span>
              </div>
            </div>
          ))}
        </div>

        {/* Example Detail Modal */}
        {selectedExample && (
          <div className="bg-white rounded-lg shadow-2xl p-8 border">
            <div className="flex items-center justify-between mb-6">
              <div>
                <h2 className="text-3xl font-bold mb-2">{selectedExample.title}</h2>
                <p className="text-gray-600">{selectedExample.description}</p>
              </div>
              <button
                onClick={() => setSelectedExample(null)}
                className="text-gray-500 hover:text-gray-700"
              >
                ✕
              </button>
            </div>

            {/* Example Info */}
            <div className="grid md:grid-cols-3 gap-4 mb-6 p-4 bg-gray-50 rounded-lg">
              <div className="flex items-center space-x-2">
                <Clock className="w-5 h-5 text-blue-600" />
                <div>
                  <div className="text-sm text-gray-600">Duration</div>
                  <div className="font-semibold">{selectedExample.time}</div>
                </div>
              </div>
              <div className="flex items-center space-x-2">
                <BarChart3 className="w-5 h-5 text-green-600" />
                <div>
                  <div className="text-sm text-gray-600">Difficulty</div>
                  <div className="font-semibold">{selectedExample.difficulty}</div>
                </div>
              </div>
              <div className="flex items-center space-x-2">
                <Target className="w-5 h-5 text-purple-600" />
                <div>
                  <div className="text-sm text-gray-600">Category</div>
                  <div className="font-semibold capitalize">{selectedExample.category}</div>
                </div>
              </div>
            </div>

            {/* Tags */}
            <div className="flex flex-wrap gap-2 mb-6">
              {selectedExample.tags.map((tag) => (
                <span key={tag} className="px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-sm">
                  {tag}
                </span>
              ))}
            </div>

            {/* Use Case */}
            <div className="mb-6 p-4 bg-blue-50 rounded-lg">
              <h3 className="font-semibold mb-2 flex items-center">
                <Target className="w-5 h-5 mr-2 text-blue-600" />
                When to Use This
              </h3>
              <p className="text-gray-700">{selectedExample.useCase}</p>
            </div>

            {/* Code */}
            <div className="mb-6">
              <h3 className="text-xl font-semibold mb-3 flex items-center">
                <Code className="w-5 h-5 mr-2" />
                Complete Code Example
              </h3>
              <CodeBlock code={selectedExample.code} language="python" />
            </div>

            {/* Explanation */}
            <div className="p-4 bg-green-50 rounded-lg">
              <h3 className="font-semibold mb-2 flex items-center">
                <Brain className="w-5 h-5 mr-2 text-green-600" />
                How It Works
              </h3>
              <p className="text-gray-700">{selectedExample.explanation}</p>
            </div>

            {/* Action Buttons */}
            <div className="flex space-x-4 mt-6 pt-6 border-t">
              <button className="flex items-center space-x-2 bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors">
                <Download className="w-4 h-4" />
                <span>Download Code</span>
              </button>
              <button className="flex items-center space-x-2 bg-green-600 text-white px-4 py-2 rounded-lg hover:bg-green-700 transition-colors">
                <PlayCircle className="w-4 h-4" />
                <span>Run Example</span>
              </button>
              <button 
                onClick={() => setSelectedExample(null)}
                className="flex items-center space-x-2 bg-gray-600 text-white px-4 py-2 rounded-lg hover:bg-gray-700 transition-colors"
              >
                <Eye className="w-4 h-4" />
                <span>Back to Examples</span>
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ExamplesPage;
