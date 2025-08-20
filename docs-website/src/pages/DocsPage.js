import React from 'react';
import { BookOpen, ArrowRight, Code, Play } from 'lucide-react';
import CodeBlock from '../components/CodeBlock';

const DocsPage = () => {
  const installationCode = `# Install from PyPI (when published)
pip install easymarl

# Or install from GitHub
pip install git+https://github.com/shreyanmitra/EasyMARL.git

# Install with all features
pip install "easymarl[all]"`;

  const basicUsageCode = `import easymarl
from easymarl.algorithms import IPPO
from easymarl.environments import make_env

# Create environment
env = make_env('MultiGrid-Empty-8x8-v0', n_agents=4)

# Create algorithm
ippo = IPPO(env, config={
    'learning_rate': 3e-4,
    'gamma': 0.99,
    'episodes': 2000
})

# Train
metrics = ippo.train()
print(f"Training completed! Mean reward: {metrics['mean_reward']}")`;

  return (
    <div className="min-h-screen pt-24 pb-20">
      <div className="max-w-7xl mx-auto px-4">
        {/* Header */}
        <div className="text-center mb-16">
          <h1 className="text-5xl md:text-6xl font-bold mb-6">
            <span className="gradient-text">Documentation</span>
          </h1>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Complete guide to using EasyMARL for multi-agent reinforcement learning.
            From basic installation to advanced research applications.
          </p>
        </div>

        {/* Quick Navigation */}
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 mb-16">
          <div className="bg-white p-6 rounded-lg shadow-lg border border-gray-100 hover:shadow-xl transition-all duration-300">
            <BookOpen className="w-8 h-8 text-blue-500 mb-4" />
            <h3 className="text-lg font-semibold mb-2">Installation</h3>
            <p className="text-gray-600 text-sm mb-4">Get started with EasyMARL installation and setup.</p>
            <button className="text-blue-500 font-medium text-sm flex items-center space-x-1">
              <span>Learn more</span>
              <ArrowRight className="w-4 h-4" />
            </button>
          </div>

          <div className="bg-white p-6 rounded-lg shadow-lg border border-gray-100 hover:shadow-xl transition-all duration-300">
            <Code className="w-8 h-8 text-purple-500 mb-4" />
            <h3 className="text-lg font-semibold mb-2">API Reference</h3>
            <p className="text-gray-600 text-sm mb-4">Detailed documentation of all classes and functions.</p>
            <button className="text-purple-500 font-medium text-sm flex items-center space-x-1">
              <span>Explore API</span>
              <ArrowRight className="w-4 h-4" />
            </button>
          </div>

          <div className="bg-white p-6 rounded-lg shadow-lg border border-gray-100 hover:shadow-xl transition-all duration-300">
            <Play className="w-8 h-8 text-green-500 mb-4" />
            <h3 className="text-lg font-semibold mb-2">Tutorials</h3>
            <p className="text-gray-600 text-sm mb-4">Step-by-step guides for learning MARL concepts.</p>
            <button className="text-green-500 font-medium text-sm flex items-center space-x-1">
              <span>Start learning</span>
              <ArrowRight className="w-4 h-4" />
            </button>
          </div>

          <div className="bg-white p-6 rounded-lg shadow-lg border border-gray-100 hover:shadow-xl transition-all duration-300">
            <Code className="w-8 h-8 text-orange-500 mb-4" />
            <h3 className="text-lg font-semibold mb-2">Examples</h3>
            <p className="text-gray-600 text-sm mb-4">Code examples and implementation patterns.</p>
            <button className="text-orange-500 font-medium text-sm flex items-center space-x-1">
              <span>View examples</span>
              <ArrowRight className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Main Content */}
        <div className="grid lg:grid-cols-2 gap-12">
          {/* Installation */}
          <div className="bg-white p-8 rounded-lg shadow-lg">
            <h2 className="text-3xl font-bold mb-6">🚀 Quick Installation</h2>
            <p className="text-gray-600 mb-6">
              Get EasyMARL up and running in minutes. Choose the installation method 
              that works best for your use case.
            </p>
            <CodeBlock code={installationCode} language="bash" title="Installation Commands" />
            
            <div className="mt-6 p-4 bg-blue-50 rounded-lg border border-blue-200">
              <h4 className="font-semibold text-blue-800 mb-2">💡 Pro Tip</h4>
              <p className="text-blue-700 text-sm">
                Use <code className="bg-blue-100 px-2 py-1 rounded">pip install "easymarl[all]"</code> to 
                install all optional dependencies including GUI, tracking, and optimization features.
              </p>
            </div>
          </div>

          {/* Basic Usage */}
          <div className="bg-white p-8 rounded-lg shadow-lg">
            <h2 className="text-3xl font-bold mb-6">🎯 Basic Usage</h2>
            <p className="text-gray-600 mb-6">
              Train your first multi-agent reinforcement learning algorithm with just 
              a few lines of code.
            </p>
            <CodeBlock code={basicUsageCode} language="python" title="Basic Usage Example" />
            
            <div className="mt-6 p-4 bg-green-50 rounded-lg border border-green-200">
              <h4 className="font-semibold text-green-800 mb-2">🎉 Success!</h4>
              <p className="text-green-700 text-sm">
                This example trains 4 IPPO agents in a MultiGrid environment. 
                The training progress will be displayed in real-time.
              </p>
            </div>
          </div>
        </div>

        {/* Feature Sections */}
        <div className="mt-16 space-y-12">
          <div className="bg-gradient-to-r from-blue-50 to-purple-50 p-8 rounded-lg">
            <h2 className="text-3xl font-bold mb-6">📚 Documentation Sections</h2>
            <div className="grid md:grid-cols-3 gap-6">
              <div>
                <h3 className="text-xl font-semibold mb-3">Algorithms</h3>
                <ul className="space-y-2 text-gray-600">
                  <li>• IPPO (Independent PPO)</li>
                  <li>• QMIX (Value Factorization)</li>
                  <li>• MADDPG (Multi-Agent DDPG)</li>
                  <li>• MAPPO (Multi-Agent PPO)</li>
                  <li>• VDN, COMA, and more...</li>
                </ul>
              </div>
              <div>
                <h3 className="text-xl font-semibold mb-3">Environments</h3>
                <ul className="space-y-2 text-gray-600">
                  <li>• MultiGrid environments</li>
                  <li>• Vectorized training</li>
                  <li>• Custom environment creation</li>
                  <li>• Environment wrappers</li>
                  <li>• Observation preprocessing</li>
                </ul>
              </div>
              <div>
                <h3 className="text-xl font-semibold mb-3">Advanced Features</h3>
                <ul className="space-y-2 text-gray-600">
                  <li>• Experiment tracking (W&B)</li>
                  <li>• Hyperparameter tuning</li>
                  <li>• Model saving/loading</li>
                  <li>• Performance optimization</li>
                  <li>• Custom neural networks</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DocsPage;
