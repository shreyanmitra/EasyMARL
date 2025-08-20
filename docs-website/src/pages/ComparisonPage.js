import React from 'react';
import { 
  CheckCircle, 
  XCircle, 
  AlertCircle, 
  Star, 
  Users, 
  Code, 
  Zap, 
  BookOpen,
  Monitor,
  Settings,
  Layers,
  Target
} from 'lucide-react';

const ComparisonPage = () => {
  
  const libraries = [
    {
      name: 'EasyMARL',
      description: 'Educational MARL framework with 21+ algorithms and GUI interface',
      category: 'Educational & Research',
      languages: ['Python'],
      website: 'https://github.com/shreyanmitra/EasyMARL',
      strengths: [
        'Educational focus with detailed documentation',
        '21+ implemented algorithms in unified interface',
        'Web-based GUI for beginners',
        'Custom environment builder',
        'Consistent API across all algorithms',
        'MultiGrid environment specialization',
        'Custom neural network architectures',
        'Real-time training visualization'
      ],
      weaknesses: [
        'Limited to discrete action spaces',
        'Focused on grid-world environments',
        'Newer framework with smaller community'
      ],
      algorithms: 21,
      environments: '16+ MultiGrid variants',
      gui: true,
      educational: true,
      production: false,
      customEnvs: true,
      documentation: 'Excellent',
      communitySize: 'Small',
      lastUpdate: '2024'
    },
    {
      name: 'Ray RLlib',
      description: 'Production-grade distributed RL library with MARL support',
      category: 'Production & Research',
      languages: ['Python'],
      website: 'https://docs.ray.io/en/latest/rllib/',
      strengths: [
        'Production-ready with distributed training',
        'High performance and scalability',
        'Extensive algorithm library',
        'Strong continuous action support',
        'Integration with Ray ecosystem',
        'Hyperparameter tuning with Tune',
        'Commercial support available'
      ],
      weaknesses: [
        'Steep learning curve for beginners',
        'Complex setup and configuration',
        'MARL features require expertise',
        'Limited educational resources',
        'No built-in GUI interface'
      ],
      algorithms: 30,
      environments: 'Gym, custom, many formats',
      gui: false,
      educational: false,
      production: true,
      customEnvs: true,
      documentation: 'Good',
      communitySize: 'Large',
      lastUpdate: '2024'
    },
    {
      name: 'PettingZoo + Stable-Baselines3',
      description: 'MARL environment standard with popular RL library',
      category: 'Research & Development',
      languages: ['Python'],
      website: 'https://pettingzoo.farama.org/',
      strengths: [
        'Standard MARL environment API',
        'Large collection of environments',
        'Good integration with SB3',
        'Active development community',
        'Wide variety of domains',
        'Good documentation'
      ],
      weaknesses: [
        'Requires combining multiple libraries',
        'No unified MARL algorithm interface',
        'Limited built-in MARL algorithms',
        'No GUI interface',
        'Setup complexity for beginners'
      ],
      algorithms: 10,
      environments: '50+ across domains',
      gui: false,
      educational: false,
      production: false,
      customEnvs: true,
      documentation: 'Good',
      communitySize: 'Medium',
      lastUpdate: '2024'
    },
    {
      name: 'EPyMARL',
      description: 'Research-focused MARL framework from Oxford',
      category: 'Research',
      languages: ['Python'],
      website: 'https://github.com/oxwhirl/epymarl',
      strengths: [
        'Research-proven algorithms',
        'Good algorithm implementations',
        'StarCraft II integration',
        'Academic credibility',
        'Clean codebase structure'
      ],
      weaknesses: [
        'Limited documentation for beginners',
        'Research-only focus',
        'No GUI interface',
        'Limited environment support',
        'Complex setup process'
      ],
      algorithms: 15,
      environments: 'SMAC, custom',
      gui: false,
      educational: false,
      production: false,
      customEnvs: false,
      documentation: 'Limited',
      communitySize: 'Small',
      lastUpdate: '2023'
    },
    {
      name: 'MAVA',
      description: 'JAX-based MARL framework by DeepMind',
      category: 'Research',
      languages: ['Python', 'JAX'],
      website: 'https://github.com/deepmind/mava',
      strengths: [
        'High-performance JAX implementation',
        'Modern functional programming approach',
        'DeepMind backing',
        'Cutting-edge algorithms',
        'GPU/TPU optimization'
      ],
      weaknesses: [
        'Requires JAX expertise',
        'Limited beginner resources',
        'Complex for educational use',
        'No GUI interface',
        'Small user community'
      ],
      algorithms: 8,
      environments: 'JAX-compatible',
      gui: false,
      educational: false,
      production: false,
      customEnvs: true,
      documentation: 'Good',
      communitySize: 'Small',
      lastUpdate: '2024'
    },
    {
      name: 'MARL-Algorithms',
      description: 'Collection of MARL algorithm implementations',
      category: 'Educational',
      languages: ['Python'],
      website: 'https://github.com/marlbenchmark/on-policy',
      strengths: [
        'Many algorithm implementations',
        'Good for learning algorithm details',
        'Research reproducibility focus',
        'Clean implementation code'
      ],
      weaknesses: [
        'No unified interface',
        'Limited documentation',
        'Inconsistent API across algorithms',
        'No GUI or visualization tools',
        'Requires manual setup for each algorithm'
      ],
      algorithms: 12,
      environments: 'Various',
      gui: false,
      educational: true,
      production: false,
      customEnvs: false,
      documentation: 'Limited',
      communitySize: 'Small',
      lastUpdate: '2023'
    }
  ];

  const criteria = [
    { key: 'educational', label: 'Educational Focus', icon: <BookOpen className="w-4 h-4" /> },
    { key: 'gui', label: 'GUI Interface', icon: <Monitor className="w-4 h-4" /> },
    { key: 'customEnvs', label: 'Custom Environments', icon: <Settings className="w-4 h-4" /> },
    { key: 'production', label: 'Production Ready', icon: <Zap className="w-4 h-4" /> }
  ];

  const getIcon = (value, criterion) => {
    if (value === true) return <CheckCircle className="w-5 h-5 text-green-500" />;
    if (value === false) return <XCircle className="w-5 h-5 text-red-500" />;
    return <AlertCircle className="w-5 h-5 text-yellow-500" />;
  };

  const getDocumentationColor = (level) => {
    switch(level) {
      case 'Excellent': return 'text-green-600 bg-green-100';
      case 'Good': return 'text-blue-600 bg-blue-100';
      case 'Limited': return 'text-yellow-600 bg-yellow-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getCommunityColor = (size) => {
    switch(size) {
      case 'Large': return 'text-green-600 bg-green-100';
      case 'Medium': return 'text-blue-600 bg-blue-100';
      case 'Small': return 'text-gray-600 bg-gray-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="container mx-auto px-6 py-12">
        
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            📊 MARL Library Comparison
          </h1>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Compare EasyMARL with other Multi-Agent Reinforcement Learning frameworks. 
            Find the right tool for your research, education, or production needs.
          </p>
        </div>

        {/* Quick Comparison Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-12">
          {libraries.map((library, index) => (
            <div key={index} className="bg-white rounded-lg shadow-lg p-6 hover:shadow-xl transition-all duration-300">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-xl font-bold text-gray-900">{library.name}</h3>
                <span className="px-3 py-1 text-sm bg-blue-100 text-blue-800 rounded-full">
                  {library.category}
                </span>
              </div>
              
              <p className="text-gray-600 mb-4 text-sm">{library.description}</p>
              
              {/* Key Metrics */}
              <div className="grid grid-cols-2 gap-3 mb-4">
                <div className="text-center p-2 bg-gray-50 rounded">
                  <div className="text-lg font-bold text-blue-600">{library.algorithms}</div>
                  <div className="text-xs text-gray-500">Algorithms</div>
                </div>
                <div className="text-center p-2 bg-gray-50 rounded">
                  <div className="text-lg font-bold text-green-600">
                    {library.environments.split(' ')[0]}
                  </div>
                  <div className="text-xs text-gray-500">Environments</div>
                </div>
              </div>

              {/* Feature Icons */}
              <div className="flex justify-center space-x-4 mb-4">
                {criteria.map((criterion) => (
                  <div key={criterion.key} className="flex flex-col items-center">
                    {getIcon(library[criterion.key], criterion.key)}
                    <span className="text-xs text-gray-500 mt-1">{criterion.label.split(' ')[0]}</span>
                  </div>
                ))}
              </div>

              {/* Documentation & Community */}
              <div className="flex justify-between text-xs">
                <span className={`px-2 py-1 rounded ${getDocumentationColor(library.documentation)}`}>
                  {library.documentation} Docs
                </span>
                <span className={`px-2 py-1 rounded ${getCommunityColor(library.communitySize)}`}>
                  {library.communitySize} Community
                </span>
              </div>
            </div>
          ))}
        </div>

        {/* Detailed Comparison Table */}
        <div className="bg-white rounded-lg shadow-lg overflow-hidden mb-12">
          <div className="px-6 py-4 bg-gray-50 border-b">
            <h2 className="text-2xl font-bold text-gray-900">📋 Detailed Feature Comparison</h2>
          </div>
          
          <div className="overflow-x-auto">
            <table className="min-w-full">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Library
                  </th>
                  <th className="px-6 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Algorithms
                  </th>
                  <th className="px-6 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Environments
                  </th>
                  <th className="px-6 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">
                    GUI
                  </th>
                  <th className="px-6 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Educational
                  </th>
                  <th className="px-6 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Production
                  </th>
                  <th className="px-6 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Documentation
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {libraries.map((library, index) => (
                  <tr key={index} className={index === 0 ? 'bg-blue-50' : 'hover:bg-gray-50'}>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center">
                        {index === 0 && <Star className="w-4 h-4 text-yellow-500 mr-2" />}
                        <div>
                          <div className="text-sm font-medium text-gray-900">{library.name}</div>
                          <div className="text-sm text-gray-500">{library.category}</div>
                        </div>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-center text-sm text-gray-900">
                      <span className="font-semibold">{library.algorithms}</span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-center text-sm text-gray-900">
                      {library.environments}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-center">
                      {getIcon(library.gui)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-center">
                      {getIcon(library.educational)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-center">
                      {getIcon(library.production)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-center">
                      <span className={`px-2 py-1 text-xs rounded ${getDocumentationColor(library.documentation)}`}>
                        {library.documentation}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Use Case Recommendations */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-12">
          <div className="bg-white rounded-lg shadow-lg p-6">
            <div className="flex items-center mb-4">
              <BookOpen className="w-8 h-8 text-blue-500 mr-3" />
              <h3 className="text-xl font-bold text-gray-900">For Education</h3>
            </div>
            <p className="text-gray-600 mb-4">
              Learning MARL concepts and experimenting with algorithms
            </p>
            <div className="space-y-2">
              <div className="flex items-center">
                <Star className="w-4 h-4 text-yellow-500 mr-2" />
                <span className="font-semibold text-blue-600">EasyMARL</span>
                <span className="ml-2 text-sm text-gray-500">- Best for beginners</span>
              </div>
              <div className="flex items-center">
                <span className="font-semibold text-gray-600 ml-6">MARL-Algorithms</span>
                <span className="ml-2 text-sm text-gray-500">- Algorithm studies</span>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow-lg p-6">
            <div className="flex items-center mb-4">
              <Code className="w-8 h-8 text-green-500 mr-3" />
              <h3 className="text-xl font-bold text-gray-900">For Research</h3>
            </div>
            <p className="text-gray-600 mb-4">
              Prototyping new algorithms and running experiments
            </p>
            <div className="space-y-2">
              <div className="flex items-center">
                <Star className="w-4 h-4 text-yellow-500 mr-2" />
                <span className="font-semibold text-green-600">EPyMARL</span>
                <span className="ml-2 text-sm text-gray-500">- Academic proven</span>
              </div>
              <div className="flex items-center">
                <span className="font-semibold text-gray-600 ml-6">PettingZoo + SB3</span>
                <span className="ml-2 text-sm text-gray-500">- Flexible</span>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow-lg p-6">
            <div className="flex items-center mb-4">
              <Zap className="w-8 h-8 text-purple-500 mr-3" />
              <h3 className="text-xl font-bold text-gray-900">For Production</h3>
            </div>
            <p className="text-gray-600 mb-4">
              Deploying MARL systems at scale in real applications
            </p>
            <div className="space-y-2">
              <div className="flex items-center">
                <Star className="w-4 h-4 text-yellow-500 mr-2" />
                <span className="font-semibold text-purple-600">Ray RLlib</span>
                <span className="ml-2 text-sm text-gray-500">- Industry standard</span>
              </div>
              <div className="flex items-center">
                <span className="font-semibold text-gray-600 ml-6">MAVA</span>
                <span className="ml-2 text-sm text-gray-500">- High performance</span>
              </div>
            </div>
          </div>
        </div>

        {/* EasyMARL Advantages */}
        <div className="bg-white rounded-lg shadow-lg p-8 mb-12">
          <h2 className="text-2xl font-bold text-gray-900 mb-6 flex items-center">
            <Star className="w-6 h-6 text-yellow-500 mr-3" />
            Why Choose EasyMARL?
          </h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h3 className="text-lg font-semibold text-gray-900 mb-3">🎓 Educational Excellence</h3>
              <ul className="space-y-2 text-gray-600">
                <li>• Beginner-friendly GUI interface</li>
                <li>• Comprehensive tutorials and documentation</li>
                <li>• Consistent API across all algorithms</li>
                <li>• Real-time training visualization</li>
                <li>• Built-in environment builder</li>
              </ul>
            </div>
            
            <div>
              <h3 className="text-lg font-semibold text-gray-900 mb-3">🔧 Research Features</h3>
              <ul className="space-y-2 text-gray-600">
                <li>• 21+ implemented algorithms</li>
                <li>• Custom neural network architectures</li>
                <li>• Unified experiment tracking</li>
                <li>• Easy algorithm comparison</li>
                <li>• Reproducible research setup</li>
              </ul>
            </div>
          </div>
          
          <div className="mt-6 p-4 bg-blue-50 rounded-lg">
            <p className="text-blue-800">
              <strong>Perfect for:</strong> Students learning MARL, researchers prototyping algorithms on discrete environments, 
              educators teaching multi-agent systems, and practitioners exploring MARL for grid-world applications.
            </p>
          </div>
        </div>

        {/* Migration Guide */}
        <div className="bg-white rounded-lg shadow-lg p-8">
          <h2 className="text-2xl font-bold text-gray-900 mb-6">🔄 Migration Guide</h2>
          
          <div className="space-y-6">
            <div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">From Ray RLlib</h3>
              <p className="text-gray-600 mb-2">
                If you're coming from RLlib and want a simpler, more educational approach:
              </p>
              <div className="bg-gray-50 p-4 rounded-lg">
                <code className="text-sm">
                  # RLlib approach (complex)<br/>
                  config = PPOConfig().environment("MyEnv").multi_agent(...)<br/>
                  <br/>
                  # EasyMARL approach (simple)<br/>
                  controller = UnifiedMultiAgentController()<br/>
                  controller.train(env_name="MyEnv", algorithm="MAPPO")
                </code>
              </div>
            </div>
            
            <div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">From PettingZoo + SB3</h3>
              <p className="text-gray-600 mb-2">
                EasyMARL provides built-in MARL algorithms instead of adapting single-agent ones:
              </p>
              <div className="bg-gray-50 p-4 rounded-lg">
                <code className="text-sm">
                  # No need for complex wrappers<br/>
                  # EasyMARL handles multi-agent environments natively<br/>
                  controller.train(env_name="MultiGrid-DoorKey-8x8", algorithm="QMIX")
                </code>
              </div>
            </div>
          </div>
        </div>

      </div>
    </div>
  );
};

export default ComparisonPage;
