import React from 'react';
import { Link } from 'react-router-dom';
import { 
  ArrowRight, 
  BookOpen, 
  Zap, 
  Users, 
  Code, 
  Star,
  TrendingUp,
  Shield,
  Globe
} from 'lucide-react';
import CodeBlock from '../components/CodeBlock';

const HomePage = () => {
  const features = [
    {
      icon: Zap,
      title: '20+ MARL Algorithms',
      description: 'From IPPO to QMIX, MADDPG to MAPPO - all the state-of-the-art algorithms implemented and ready to use.',
      gradient: 'from-blue-500 to-cyan-500'
    },
    {
      icon: Users,
      title: 'Beginner-Friendly',
      description: 'Start with simple demos, progress to research-grade implementations. Perfect for learning MARL.',
      gradient: 'from-purple-500 to-pink-500'
    },
    {
      icon: Code,
      title: 'Research-Ready',
      description: 'Unified controller with 8x vectorized speedup, experiment tracking, performance monitoring, and educational features.',
      gradient: 'from-green-500 to-teal-500'
    },
    {
      icon: Globe,
      title: 'Multiple Interfaces',
      description: 'Web GUI for beginners, Python API for developers, and full customization for researchers.',
      gradient: 'from-orange-500 to-red-500'
    },
    {
      icon: Shield,
      title: 'Production-Tested',
      description: 'Thoroughly tested algorithms with comprehensive documentation and example implementations.',
      gradient: 'from-indigo-500 to-purple-500'
    },
    {
      icon: TrendingUp,
      title: 'High Performance',
      description: 'Vectorized environments, efficient neural networks, and optimized training loops for speed.',
      gradient: 'from-pink-500 to-rose-500'
    }
  ];

  const algorithms = [
    { name: 'IPPO', type: 'Policy-Based', difficulty: 'Beginner', description: 'Independent Proximal Policy Optimization' },
    { name: 'QMIX', type: 'Value-Based', difficulty: 'Intermediate', description: 'Monotonic Value Function Factorization' },
    { name: 'MADDPG', type: 'Actor-Critic', difficulty: 'Advanced', description: 'Multi-Agent Deep Deterministic Policy Gradient' },
    { name: 'MAPPO', type: 'Policy-Based', difficulty: 'Intermediate', description: 'Multi-Agent Proximal Policy Optimization' },
    { name: 'VDN', type: 'Value-Based', difficulty: 'Beginner', description: 'Value Decomposition Networks' },
    { name: 'COMA', type: 'Actor-Critic', difficulty: 'Advanced', description: 'Counterfactual Multi-Agent Policy Gradients' }
  ];

  const quickStartCode = `# Install EasyMARL
pip install easymarl

# Quick start - 3 lines of code!
import easymarl

trainer = easymarl.QuickStart(
    algorithm='ippo',
    environment='MultiGrid-Empty-8x8-v0',
    episodes=1000
)

results = trainer.train()
print(f"Training completed! Reward: {results['mean_reward']}")`;

  const advancedCode = `from easymarl.algorithms import IPPO
from easymarl.environments import make_env
from easymarl.controllers import UnifiedMultiAgentController

# Create environment and algorithm with vectorized training
env = make_env('MultiGrid-Empty-8x8-v0', n_agents=4)

# Unified controller with 8x performance boost
controller = UnifiedMultiAgentController(
    env_name='MultiGrid-Empty-8x8-v0',
    algorithm='ippo',
    n_envs=8,  # 8 parallel environments for speedup
    enable_advanced_tracking=True,
    config={'learning_rate': 3e-4}
)

# Train with vectorized environments and full tracking
results = controller.train(total_episodes=2000)
eval_results = controller.evaluate(num_episodes=10)

print(f"Training completed! Final reward: {eval_results['average_reward']:.3f}")
print(f"8x speedup achieved with vectorized environments!")`;

  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="relative hero-pattern pt-24 pb-20 px-4">
        <div className="max-w-7xl mx-auto text-center">
          <div>
            <h1 className="text-5xl md:text-7xl font-bold mb-6">
              <span className="gradient-text">EasyMARL</span>
            </h1>
            <p className="text-xl md:text-2xl text-gray-600 mb-8 max-w-3xl mx-auto">
              The most comprehensive Multi-Agent Reinforcement Learning framework.
              <br />
              <span className="font-semibold text-gray-800">Learn, Research, and Deploy MARL algorithms with ease.</span>
            </p>
            
            <div className="flex flex-col sm:flex-row gap-4 justify-center items-center mb-12">
              <Link
                to="/docs"
                className="flex items-center space-x-2 px-8 py-4 bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-lg font-semibold hover:shadow-xl transform hover:scale-105 transition-all duration-300"
              >
                <BookOpen className="w-5 h-5" />
                <span>Get Started</span>
                <ArrowRight className="w-5 h-5" />
              </Link>
              
              <Link
                to="/algorithms"
                className="flex items-center space-x-2 px-8 py-4 bg-white text-gray-700 rounded-lg font-semibold border-2 border-gray-200 hover:border-purple-300 hover:shadow-lg transform hover:scale-105 transition-all duration-300"
              >
                <Zap className="w-5 h-5" />
                <span>View Algorithms</span>
              </Link>
            </div>

            {/* Quick Stats */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-8 max-w-4xl mx-auto">
              {[
                { number: '20+', label: 'MARL Algorithms' },
                { number: '3', label: 'Deployment Methods' },
                { number: '12+', label: 'Environments' },
                { number: '1', label: 'Core Maintainer' }
              ].map((stat, index) => (
                <div key={index} className="text-center">
                  <div className="text-3xl md:text-4xl font-bold gradient-text">{stat.number}</div>
                  <div className="text-gray-600 font-medium">{stat.label}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* Quick Start Section */}
      <section className="py-20 bg-white">
        <div className="max-w-7xl mx-auto px-4">
          <div className="text-center mb-16">
            <h2 className="text-4xl md:text-5xl font-bold mb-6">
              Get Started in <span className="gradient-text">Minutes</span>
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              From installation to training your first MARL algorithm in just a few lines of code.
            </p>
          </div>

          <div className="grid md:grid-cols-2 gap-12 items-center">
            <div>
              <h3 className="text-2xl font-bold mb-4">Beginner-Friendly</h3>
              <p className="text-gray-600 mb-6">
                Perfect for students and researchers new to MARL. Start training 
                sophisticated multi-agent algorithms with minimal setup.
              </p>
              <CodeBlock code={quickStartCode} language="python" />
            </div>

            <div>
              <h3 className="text-2xl font-bold mb-4">Research-Grade</h3>
              <p className="text-gray-600 mb-6">
                Unified controller combining educational clarity, advanced tracking, and 
                8x vectorized performance. Perfect for serious MARL research with full 
                experiment management and professional logging.
              </p>
              <CodeBlock code={advancedCode} language="python" />
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4">
          <div className="text-center mb-16">
            <h2 className="text-4xl md:text-5xl font-bold mb-6">
              Why Choose <span className="gradient-text">EasyMARL</span>?
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Built by researchers, for researchers. EasyMARL bridges the gap between 
              academic research and practical implementation.
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            {features.map((feature, index) => {
              const Icon = feature.icon;
              return (
                <div
                  key={index}
                  className="feature-card bg-white p-8 rounded-xl"
                >
                  <div className={`w-12 h-12 bg-gradient-to-r ${feature.gradient} rounded-lg flex items-center justify-center mb-6`}>
                    <Icon className="w-6 h-6 text-white" />
                  </div>
                  <h3 className="text-xl font-bold mb-4">{feature.title}</h3>
                  <p className="text-gray-600">{feature.description}</p>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {/* Algorithms Preview */}
      <section className="py-20 bg-white">
        <div className="max-w-7xl mx-auto px-4">
          <div className="text-center mb-16">
            <h2 className="text-4xl md:text-5xl font-bold mb-6">
              <span className="gradient-text">20+</span> State-of-the-Art Algorithms
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              From beginner-friendly IPPO to advanced QMIX and MADDPG. 
              All algorithms are implemented with the latest research insights.
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {algorithms.map((algo, index) => (
              <div
                key={index}
                className="bg-gray-50 p-6 rounded-lg border-2 border-transparent hover:border-purple-200 transition-all duration-300"
              >
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-xl font-bold">{algo.name}</h3>
                  <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                    algo.difficulty === 'Beginner' 
                      ? 'bg-green-100 text-green-700' 
                      : algo.difficulty === 'Intermediate'
                      ? 'bg-yellow-100 text-yellow-700'
                      : 'bg-red-100 text-red-700'
                  }`}>
                    {algo.difficulty}
                  </span>
                </div>
                <p className="text-gray-600 mb-3">{algo.description}</p>
                <div className="text-sm text-purple-600 font-medium">{algo.type}</div>
              </div>
            ))}
          </div>

          <div className="text-center mt-12">
            <Link
              to="/algorithms"
              className="inline-flex items-center space-x-2 px-8 py-4 bg-gradient-to-r from-purple-500 to-pink-500 text-white rounded-lg font-semibold hover:shadow-xl transform hover:scale-105 transition-all duration-300"
            >
              <span>Explore All Algorithms</span>
              <ArrowRight className="w-5 h-5" />
            </Link>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 bg-gradient-to-r from-blue-500 to-purple-600 text-white">
        <div className="max-w-4xl mx-auto text-center px-4">
          <div>
            <h2 className="text-4xl md:text-5xl font-bold mb-6">
              Ready to Start Your MARL Journey?
            </h2>
            <p className="text-xl mb-8 opacity-90">
              Join the growing community of researchers and developers using EasyMARL to advance 
              multi-agent reinforcement learning research and education.
            </p>
            
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link
                to="/docs"
                className="flex items-center space-x-2 px-8 py-4 bg-white text-purple-600 rounded-lg font-semibold hover:shadow-xl transform hover:scale-105 transition-all duration-300"
              >
                <BookOpen className="w-5 h-5" />
                <span>Read the Docs</span>
              </Link>
              
              <a
                href="https://github.com/shreyanmitra/EasyMARL"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center space-x-2 px-8 py-4 bg-transparent border-2 border-white text-white rounded-lg font-semibold hover:bg-white hover:text-purple-600 transition-all duration-300"
              >
                <Star className="w-5 h-5" />
                <span>Star on GitHub</span>
              </a>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};

export default HomePage;
