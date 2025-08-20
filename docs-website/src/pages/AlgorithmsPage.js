import React, { useState } from 'react';
import { Search, BookOpen, Code, Zap, Users, Brain, Target, TrendingUp, Filter } from 'lucide-react';
import CodeBlock from '../components/CodeBlock';

const AlgorithmsPage = () => {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [selectedDifficulty, setSelectedDifficulty] = useState('all');

  const algorithms = [
    // Policy-Based Algorithms
    {
      name: 'IPPO',
      fullName: 'Independent Proximal Policy Optimization',
      category: 'Policy-Based',
      difficulty: 'Beginner',
      description: 'Independent training of PPO agents. Perfect for learning MARL basics.',
      icon: <Users className="w-6 h-6" />,
      features: ['Simple to understand', 'Fast training', 'Good baseline'],
      useCases: ['Learning MARL', 'Baseline comparisons', 'Simple coordination'],
      paperLink: 'https://arxiv.org/abs/1707.06347',
      codeExample: `controller = UnifiedMultiAgentController(
    env_name='MultiGrid-Empty-8x8',
    algorithm='ippo',
    n_envs=8,
    educational_mode=True
)
results = controller.train(total_episodes=1000)`
    },
    {
      name: 'MAPPO',
      fullName: 'Multi-Agent Proximal Policy Optimization',
      category: 'Policy-Based',
      difficulty: 'Intermediate',
      description: 'Centralized training with decentralized execution using shared value function.',
      icon: <Brain className="w-6 h-6" />,
      features: ['Centralized training', 'Shared value function', 'Better coordination'],
      useCases: ['Cooperative tasks', 'Partial observability', 'Large action spaces'],
      paperLink: 'https://arxiv.org/abs/2103.01955',
      codeExample: `controller = UnifiedMultiAgentController(
    env_name='MultiGrid-Cooperative-8x8',
    algorithm='mappo',
    n_envs=8,
    config={'shared_critic': True}
)`
    },
    {
      name: 'MFPO',
      fullName: 'Mean Field Policy Optimization',
      category: 'Policy-Based',
      difficulty: 'Advanced',
      description: 'Handles large-scale multi-agent systems using mean field approximation.',
      icon: <TrendingUp className="w-6 h-6" />,
      features: ['Scalable to many agents', 'Mean field theory', 'Efficient computation'],
      useCases: ['Large-scale systems', 'Swarm robotics', 'Traffic control'],
      paperLink: 'https://arxiv.org/abs/1802.05438',
      codeExample: `controller = UnifiedMultiAgentController(
    env_name='LargeScale-Environment',
    algorithm='mfpo',
    n_envs=4,
    config={'n_agents': 100}
)`
    },

    // Value-Based Algorithms
    {
      name: 'VDN',
      fullName: 'Value Decomposition Networks',
      category: 'Value-Based',
      difficulty: 'Beginner',
      description: 'Decomposes team value function into individual agent value functions.',
      icon: <Target className="w-6 h-6" />,
      features: ['Value decomposition', 'Additive assumption', 'Centralized training'],
      useCases: ['Cooperative tasks', 'Shared rewards', 'Simple coordination'],
      paperLink: 'https://arxiv.org/abs/1706.05296',
      codeExample: `controller = UnifiedMultiAgentController(
    env_name='MultiGrid-Cooperative-6x6',
    algorithm='vdn',
    n_envs=8,
    config={'mixer': 'vdn'}
)`
    },
    {
      name: 'QMIX',
      fullName: 'QMIX Value Factorization',
      category: 'Value-Based',
      difficulty: 'Intermediate',
      description: 'Monotonic value factorization using mixing networks.',
      icon: <Zap className="w-6 h-6" />,
      features: ['Monotonic factorization', 'Mixing network', 'IGM guarantee'],
      useCases: ['Complex coordination', 'Cooperative games', 'StarCraft II'],
      paperLink: 'https://arxiv.org/abs/1803.11485',
      codeExample: `controller = UnifiedMultiAgentController(
    env_name='MultiGrid-Complex-12x12',
    algorithm='qmix',
    n_envs=8,
    config={'mixer': 'qmix', 'hypernet_layers': [64, 64]}
)`
    },
    {
      name: 'QTRAN',
      fullName: 'QTRAN Transformation',
      category: 'Value-Based',
      difficulty: 'Advanced',
      description: 'General value factorization without monotonicity constraints.',
      icon: <Brain className="w-6 h-6" />,
      features: ['General factorization', 'No monotonicity', 'Counterfactual reasoning'],
      useCases: ['Non-monotonic tasks', 'Complex interactions', 'Advanced coordination'],
      paperLink: 'https://arxiv.org/abs/1905.05408',
      codeExample: `controller = UnifiedMultiAgentController(
    env_name='MultiGrid-NonMonotonic-10x10',
    algorithm='qtran',
    n_envs=4,
    config={'qtran_type': 'qtran_base'}
)`
    },

    // Actor-Critic Algorithms
    {
      name: 'MADDPG',
      fullName: 'Multi-Agent Deep Deterministic Policy Gradient',
      category: 'Actor-Critic',
      difficulty: 'Intermediate',
      description: 'Centralized training with decentralized execution for continuous actions.',
      icon: <Target className="w-6 h-6" />,
      features: ['Continuous actions', 'Centralized critics', 'Policy gradients'],
      useCases: ['Continuous control', 'Robotics', 'Physical simulations'],
      paperLink: 'https://arxiv.org/abs/1706.02275',
      codeExample: `controller = UnifiedMultiAgentController(
    env_name='Continuous-MultiAgent-Env',
    algorithm='maddpg',
    n_envs=4,
    config={'tau': 0.01, 'lr_actor': 1e-4, 'lr_critic': 1e-3}
)`
    },
    {
      name: 'COMA',
      fullName: 'Counterfactual Multi-Agent Policy Gradients',
      category: 'Actor-Critic',
      difficulty: 'Advanced',
      description: 'Addresses credit assignment using counterfactual baselines.',
      icon: <Brain className="w-6 h-6" />,
      features: ['Counterfactual reasoning', 'Credit assignment', 'Centralized critic'],
      useCases: ['Credit assignment', 'Complex coordination', 'StarCraft scenarios'],
      paperLink: 'https://arxiv.org/abs/1705.08926',
      codeExample: `controller = UnifiedMultiAgentController(
    env_name='MultiGrid-CreditAssignment-10x10',
    algorithm='coma',
    n_envs=4,
    config={'lambda_gae': 0.95, 'counterfactual': True}
)`
    },
    {
      name: 'MAACC',
      fullName: 'Multi-Agent Actor-Critic with Communication',
      category: 'Actor-Critic',
      difficulty: 'Advanced',
      description: 'Actor-critic with explicit communication channels between agents.',
      icon: <Users className="w-6 h-6" />,
      features: ['Communication channels', 'Message passing', 'Coordination'],
      useCases: ['Communication tasks', 'Information sharing', 'Cooperative planning'],
      paperLink: 'https://arxiv.org/abs/1810.02912',
      codeExample: `controller = UnifiedMultiAgentController(
    env_name='MultiGrid-Communication-8x8',
    algorithm='maacc',
    n_envs=4,
    config={'comm_size': 32, 'comm_type': 'continuous'}
)`
    }
  ];

  const categories = ['all', 'Policy-Based', 'Value-Based', 'Actor-Critic'];
  const difficulties = ['all', 'Beginner', 'Intermediate', 'Advanced'];

  const filteredAlgorithms = algorithms.filter(algo => {
    const matchesSearch = algo.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         algo.fullName.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesCategory = selectedCategory === 'all' || algo.category === selectedCategory;
    const matchesDifficulty = selectedDifficulty === 'all' || algo.difficulty === selectedDifficulty;
    return matchesSearch && matchesCategory && matchesDifficulty;
  });

  const getDifficultyColor = (difficulty) => {
    switch(difficulty) {
      case 'Beginner': return 'bg-green-100 text-green-700 border-green-200';
      case 'Intermediate': return 'bg-yellow-100 text-yellow-700 border-yellow-200';
      case 'Advanced': return 'bg-red-100 text-red-700 border-red-200';
      default: return 'bg-gray-100 text-gray-700 border-gray-200';
    }
  };

  const getCategoryColor = (category) => {
    switch(category) {
      case 'Policy-Based': return 'bg-blue-100 text-blue-700';
      case 'Value-Based': return 'bg-purple-100 text-purple-700';
      case 'Actor-Critic': return 'bg-orange-100 text-orange-700';
      default: return 'bg-gray-100 text-gray-700';
    }
  };

  return (
    <div className="min-h-screen pt-24 pb-20 bg-gray-50">
      <div className="max-w-7xl mx-auto px-4">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-5xl md:text-6xl font-bold mb-6">
            MARL <span className="gradient-text">Algorithms</span>
          </h1>
          <p className="text-xl text-gray-600 max-w-4xl mx-auto mb-8">
            Comprehensive library of state-of-the-art Multi-Agent Reinforcement Learning algorithms.
            Each algorithm is thoroughly tested, documented, and ready for research or production use.
          </p>
        </div>

        {/* Search and Filters */}
        <div className="bg-white p-6 rounded-lg shadow-lg mb-8">
          <div className="flex flex-col md:flex-row gap-4 items-center">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
              <input
                type="text"
                placeholder="Search algorithms..."
                className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
              />
            </div>
            
            <div className="flex gap-4">
              <select
                className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                value={selectedCategory}
                onChange={(e) => setSelectedCategory(e.target.value)}
              >
                {categories.map(cat => (
                  <option key={cat} value={cat}>
                    {cat === 'all' ? 'All Categories' : cat}
                  </option>
                ))}
              </select>
              
              <select
                className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                value={selectedDifficulty}
                onChange={(e) => setSelectedDifficulty(e.target.value)}
              >
                {difficulties.map(diff => (
                  <option key={diff} value={diff}>
                    {diff === 'all' ? 'All Levels' : diff}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </div>

        {/* Algorithm Grid */}
        <div className="grid lg:grid-cols-2 gap-8">
          {filteredAlgorithms.map((algo, index) => (
            <div key={index} className="bg-white rounded-lg shadow-lg overflow-hidden hover:shadow-xl transition-shadow duration-300">
              {/* Header */}
              <div className="p-6 border-b border-gray-100">
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-center space-x-3">
                    <div className="p-2 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg text-white">
                      {algo.icon}
                    </div>
                    <div>
                      <h3 className="text-2xl font-bold text-gray-900">{algo.name}</h3>
                      <p className="text-sm text-gray-600">{algo.fullName}</p>
                    </div>
                  </div>
                  <div className="flex flex-col gap-2">
                    <span className={`px-3 py-1 rounded-full text-xs font-medium ${getCategoryColor(algo.category)}`}>
                      {algo.category}
                    </span>
                    <span className={`px-3 py-1 rounded-full text-xs font-medium border ${getDifficultyColor(algo.difficulty)}`}>
                      {algo.difficulty}
                    </span>
                  </div>
                </div>
                <p className="text-gray-700">{algo.description}</p>
              </div>

              {/* Features */}
              <div className="p-6 border-b border-gray-100">
                <h4 className="font-semibold text-gray-900 mb-3">Key Features</h4>
                <div className="flex flex-wrap gap-2">
                  {algo.features.map((feature, idx) => (
                    <span key={idx} className="px-3 py-1 bg-blue-50 text-blue-700 rounded-full text-sm">
                      {feature}
                    </span>
                  ))}
                </div>
              </div>

              {/* Use Cases */}
              <div className="p-6 border-b border-gray-100">
                <h4 className="font-semibold text-gray-900 mb-3">Common Use Cases</h4>
                <ul className="space-y-1">
                  {algo.useCases.map((useCase, idx) => (
                    <li key={idx} className="text-gray-600 flex items-center">
                      <span className="w-2 h-2 bg-green-500 rounded-full mr-3"></span>
                      {useCase}
                    </li>
                  ))}
                </ul>
              </div>

              {/* Code Example */}
              <div className="p-6">
                <h4 className="font-semibold text-gray-900 mb-3">Quick Start</h4>
                <CodeBlock code={algo.codeExample} language="python" />
                <div className="flex justify-between items-center mt-4">
                  <a
                    href={algo.paperLink}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center space-x-2 text-blue-600 hover:text-blue-800 text-sm"
                  >
                    <BookOpen className="w-4 h-4" />
                    <span>Read Paper</span>
                  </a>
                  <button className="flex items-center space-x-2 bg-gradient-to-r from-blue-500 to-purple-600 text-white px-4 py-2 rounded-lg hover:shadow-lg transition-all duration-300">
                    <Code className="w-4 h-4" />
                    <span>Try Example</span>
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* No Results */}
        {filteredAlgorithms.length === 0 && (
          <div className="text-center py-12">
            <Filter className="w-16 h-16 text-gray-400 mx-auto mb-4" />
            <h3 className="text-xl font-semibold text-gray-600 mb-2">No algorithms found</h3>
            <p className="text-gray-500">Try adjusting your search or filter criteria</p>
          </div>
        )}

        {/* Algorithm Comparison */}
        <div className="mt-16 bg-white rounded-lg shadow-lg p-8">
          <h2 className="text-3xl font-bold mb-6">Algorithm Comparison</h2>
          <div className="overflow-x-auto">
            <table className="min-w-full">
              <thead>
                <tr className="border-b border-gray-200">
                  <th className="text-left py-3 px-4 font-semibold">Algorithm</th>
                  <th className="text-left py-3 px-4 font-semibold">Category</th>
                  <th className="text-left py-3 px-4 font-semibold">Action Space</th>
                  <th className="text-left py-3 px-4 font-semibold">Complexity</th>
                  <th className="text-left py-3 px-4 font-semibold">Best For</th>
                </tr>
              </thead>
              <tbody>
                {algorithms.slice(0, 6).map((algo, idx) => (
                  <tr key={idx} className="border-b border-gray-100 hover:bg-gray-50">
                    <td className="py-3 px-4 font-medium">{algo.name}</td>
                    <td className="py-3 px-4">
                      <span className={`px-2 py-1 rounded text-xs ${getCategoryColor(algo.category)}`}>
                        {algo.category}
                      </span>
                    </td>
                    <td className="py-3 px-4 text-gray-600">
                      {algo.name === 'MADDPG' ? 'Continuous' : 'Discrete'}
                    </td>
                    <td className="py-3 px-4">
                      <span className={`px-2 py-1 rounded text-xs border ${getDifficultyColor(algo.difficulty)}`}>
                        {algo.difficulty}
                      </span>
                    </td>
                    <td className="py-3 px-4 text-gray-600">{algo.useCases[0]}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AlgorithmsPage;
