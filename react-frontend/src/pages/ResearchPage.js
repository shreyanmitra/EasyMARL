import React, { useState, useEffect } from 'react';
import ParameterConfigurator from '../components/ParameterConfigurator';
import { motion, AnimatePresence } from 'framer-motion';

/**
 * Research Interface Component
 * 
 * Advanced research-oriented interface for MARL algorithm discovery,
 * experimentation, and hyperparameter optimization.
 */
const ResearchPage = () => {
  const [activeTab, setActiveTab] = useState('discover');
  const [algorithms, setAlgorithms] = useState({});
  const [selectedAlgorithm, setSelectedAlgorithm] = useState('');
  const [algorithmInfo, setAlgorithmInfo] = useState(null);
  const [experimentConfig, setExperimentConfig] = useState({
    name: '',
    description: '',
    algorithm: 'ippo',
    environment: 'MultiGrid-Cluttered-Fixed-15x15',
    total_episodes: 10000,
    n_agents: 2,
    agent_configs: []
  });
  const [parameterSchema, setParameterSchema] = useState({});
  const [parameterConfig, setParameterConfig] = useState({});
  const [agentConfigs, setAgentConfigs] = useState([]);
  const [nAgents, setNAgents] = useState(2);
  const [memoryShared, setMemoryShared] = useState(false);
  const [communicationEnabled, setCommunicationEnabled] = useState(false);
  const [optimizationStatus, setOptimizationStatus] = useState(null);
  const [experiments, setExperiments] = useState([]);
  const [hyperparameterResults, setHyperparameterResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [comparisonResults, setComparisonResults] = useState(null);

  // Load algorithm browser data on component mount
  useEffect(() => {
    loadAlgorithmBrowser();
  }, []);

  // Load algorithm details when selection changes
  useEffect(() => {
    if (selectedAlgorithm) {
      loadAlgorithmDetails(selectedAlgorithm);
      loadParameterSchema(selectedAlgorithm);
    }
  }, [selectedAlgorithm]);

  const loadAlgorithmBrowser = async () => {
    try {
      const response = await fetch('/api/research/algorithms/browse');
      const data = await response.json();
      if (data.success) {
        setAlgorithms(data.data);
      }
    } catch (error) {
      console.error('Failed to load algorithms:', error);
    }
  };

  const loadAlgorithmDetails = async (algorithm) => {
    try {
      const response = await fetch(`/api/research/algorithms/${algorithm}/info`);
      const data = await response.json();
      if (data.success) {
        setAlgorithmInfo(data.data);
      }
    } catch (error) {
      console.error('Failed to load algorithm details:', error);
    }
  };

  const loadParameterSchema = async (algorithm) => {
    try {
      const response = await fetch(`/api/research/algorithms/${algorithm}/parameters`);
      const data = await response.json();
      if (data.success) {
        setParameterSchema(data.schema);
      }
    } catch (error) {
      console.error('Failed to load parameter schema:', error);
    }
  };

  const compareAlgorithms = async (selectedAlgos) => {
    try {
      const response = await fetch('/api/research/algorithms/compare', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ algorithms: selectedAlgos })
      });
      const data = await response.json();
      if (data.success) {
        setComparisonResults(data.comparison);
      }
    } catch (error) {
      console.error('Failed to compare algorithms:', error);
    }
  };

  const startHyperparameterOptimization = async () => {
    try {
      const response = await fetch('/api/research/hyperparameter-search/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          config: experimentConfig,
          search_budget: 50
        })
      });
      const data = await response.json();
      if (data.success) {
        setOptimizationStatus('running');
        // Poll for results
        pollOptimizationStatus(data.experiment_name);
      }
    } catch (error) {
      console.error('Failed to start optimization:', error);
    }
  };

  const pollOptimizationStatus = async (experimentName) => {
    const poll = async () => {
      try {
        const response = await fetch(`/api/research/hyperparameter-search/status/${experimentName}`);
        const data = await response.json();
        if (data.success) {
          if (data.status === 'completed') {
            setOptimizationStatus('completed');
            // Handle results
          } else {
            setTimeout(poll, 5000); // Poll every 5 seconds
          }
        }
      } catch (error) {
        console.error('Failed to poll optimization status:', error);
      }
    };
    poll();
  };

  const renderAlgorithmBrowser = () => (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold mb-4">🔍 Algorithm Discovery</h2>
        <p className="text-gray-600 mb-6">
          Explore and compare MARL algorithms organized by their fundamental characteristics
        </p>
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        {/* Algorithm Categories */}
        <div className="bg-white rounded-lg p-6 shadow-lg">
          <h3 className="text-lg font-semibold mb-4">📂 Browse by Category</h3>
          <div className="space-y-3">
            {Object.entries(algorithms).map(([category, algos]) => (
              <div key={category} className="border rounded-lg p-3">
                <h4 className="font-medium text-blue-600 mb-2">{category}</h4>
                <div className="flex flex-wrap gap-2">
                  {algos.map((algo) => (
                    <button
                      key={algo.name}
                      onClick={() => setSelectedAlgorithm(algo.name)}
                      className={`px-3 py-1 text-sm rounded-full border transition-colors ${
                        selectedAlgorithm === algo.name
                          ? 'bg-blue-500 text-white border-blue-500'
                          : 'bg-gray-50 text-gray-700 border-gray-300 hover:bg-blue-50'
                      }`}
                    >
                      {algo.name.toUpperCase()}
                    </button>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Algorithm Details */}
        <div className="bg-white rounded-lg p-6 shadow-lg">
          <h3 className="text-lg font-semibold mb-4">📋 Algorithm Details</h3>
          {algorithmInfo ? (
            <div className="space-y-4">
              <div>
                <h4 className="font-medium text-green-600">
                  {selectedAlgorithm.toUpperCase()}
                </h4>
                <p className="text-sm text-gray-600">{algorithmInfo.paper}</p>
              </div>
              
              <div>
                <span className="font-medium">Category:</span>
                <span className="ml-2 text-blue-600">{algorithmInfo.category}</span>
              </div>
              
              <div>
                <span className="font-medium">Complexity:</span>
                <span className={`ml-2 px-2 py-1 text-xs rounded-full ${
                  algorithmInfo.complexity === 'Beginner' ? 'bg-green-100 text-green-800' :
                  algorithmInfo.complexity === 'Intermediate' ? 'bg-yellow-100 text-yellow-800' :
                  algorithmInfo.complexity === 'Advanced' ? 'bg-orange-100 text-orange-800' :
                  'bg-red-100 text-red-800'
                }`}>
                  {algorithmInfo.complexity}
                </span>
              </div>

              <div>
                <span className="font-medium">Strengths:</span>
                <ul className="mt-2 space-y-1">
                  {algorithmInfo.strengths?.map((strength, idx) => (
                    <li key={idx} className="text-sm text-green-600 flex items-center">
                      <span className="w-2 h-2 bg-green-500 rounded-full mr-2"></span>
                      {strength}
                    </li>
                  ))}
                </ul>
              </div>

              <div>
                <span className="font-medium">Limitations:</span>
                <ul className="mt-2 space-y-1">
                  {algorithmInfo.limitations?.map((limitation, idx) => (
                    <li key={idx} className="text-sm text-red-600 flex items-center">
                      <span className="w-2 h-2 bg-red-500 rounded-full mr-2"></span>
                      {limitation}
                    </li>
                  ))}
                </ul>
              </div>

              <div>
                <span className="font-medium">Key Concepts:</span>
                <div className="mt-2 flex flex-wrap gap-2">
                  {algorithmInfo.key_concepts?.map((concept, idx) => (
                    <span key={idx} className="px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded-full">
                      {concept}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <p className="text-gray-500">Select an algorithm to view details</p>
          )}
        </div>
      </div>
    </div>
  );

  const renderExperimentDesigner = () => (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold mb-4">🧪 Experiment Designer</h2>
        <p className="text-gray-600 mb-6">
          Design and configure sophisticated MARL experiments with fine-grained parameter control
        </p>
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        {/* Experiment Configuration */}
        <div className="bg-white rounded-lg p-6 shadow-lg">
          <h3 className="text-lg font-semibold mb-4">⚙️ Experiment Setup</h3>
          
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-1">Experiment Name</label>
              <input
                type="text"
                value={experimentConfig.name}
                onChange={(e) => setExperimentConfig({...experimentConfig, name: e.target.value})}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="my_research_experiment"
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-1">Description</label>
              <textarea
                value={experimentConfig.description}
                onChange={(e) => setExperimentConfig({...experimentConfig, description: e.target.value})}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                rows="3"
                placeholder="Describe your research objective..."
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-1">Algorithm</label>
              <select
                value={experimentConfig.algorithm}
                onChange={(e) => setExperimentConfig({...experimentConfig, algorithm: e.target.value})}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                {Object.values(algorithms).flat().map((algo) => (
                  <option key={algo.name} value={algo.name}>
                    {algo.name.toUpperCase()}
                  </option>
                ))}
              </select>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-sm font-medium mb-1">Episodes</label>
                <input
                  type="number"
                  value={experimentConfig.total_episodes}
                  onChange={(e) => setExperimentConfig({...experimentConfig, total_episodes: parseInt(e.target.value)})}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Agents</label>
                <input
                  type="number"
                  value={experimentConfig.n_agents}
                  onChange={(e) => setExperimentConfig({...experimentConfig, n_agents: parseInt(e.target.value)})}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
            </div>
          </div>
        </div>

        {/* Agent Parameter Tuning */}
        <div className="bg-white rounded-lg p-6 shadow-lg">
          <h3 className="text-lg font-semibold mb-4">🎛️ Agent Parameters</h3>
          
          {parameterSchema.global_params && (
            <div className="space-y-4">
              <h4 className="font-medium text-gray-700">Global Parameters</h4>
              {Object.entries(parameterSchema.global_params).map(([param, config]) => (
                <div key={param}>
                  <label className="block text-sm font-medium mb-1">
                    {param.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                  </label>
                  <input
                    type={config.type}
                    defaultValue={config.default}
                    min={config.min}
                    max={config.max}
                    step={config.step}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    title={config.description}
                  />
                  <p className="text-xs text-gray-500 mt-1">{config.description}</p>
                </div>
              ))}
            </div>
          )}

          {parameterSchema.algorithm_params && (
            <div className="space-y-4 mt-6">
              <h4 className="font-medium text-gray-700">Algorithm-Specific Parameters</h4>
              {Object.entries(parameterSchema.algorithm_params).map(([param, config]) => (
                <div key={param}>
                  <label className="block text-sm font-medium mb-1">
                    {param.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                  </label>
                  <input
                    type={config.type}
                    defaultValue={config.default}
                    min={config.min}
                    max={config.max}
                    step={config.step}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    title={config.description}
                  />
                  <p className="text-xs text-gray-500 mt-1">{config.description}</p>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex gap-4">
        <button
          onClick={startHyperparameterOptimization}
          className="px-6 py-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors"
        >
          🔬 Start Hyperparameter Search
        </button>
        <button className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">
          🚀 Start Training
        </button>
        <button className="px-6 py-3 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors">
          💾 Save Configuration
        </button>
      </div>
    </div>
  );

  const renderComparison = () => (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold mb-4">📊 Algorithm Comparison</h2>
        <p className="text-gray-600 mb-6">
          Compare multiple algorithms across key research dimensions
        </p>
      </div>

      {/* Algorithm Selection */}
      <div className="bg-white rounded-lg p-6 shadow-lg">
        <h3 className="text-lg font-semibold mb-4">Select Algorithms to Compare</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {Object.values(algorithms).flat().map((algo) => (
            <label key={algo.name} className="flex items-center space-x-2">
              <input type="checkbox" className="rounded" />
              <span className="text-sm">{algo.name.toUpperCase()}</span>
            </label>
          ))}
        </div>
        <button
          onClick={() => compareAlgorithms(['qmix', 'vdn', 'ippo'])} // Example
          className="mt-4 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
        >
          Compare Selected
        </button>
      </div>

      {/* Comparison Results */}
      {comparisonResults && (
        <div className="bg-white rounded-lg p-6 shadow-lg">
          <h3 className="text-lg font-semibold mb-4">Comparison Results</h3>
          <div className="overflow-x-auto">
            <table className="w-full border-collapse">
              <thead>
                <tr className="border-b">
                  <th className="text-left p-2">Algorithm</th>
                  <th className="text-left p-2">Category</th>
                  <th className="text-left p-2">Complexity</th>
                  <th className="text-left p-2">Sample Efficiency</th>
                  <th className="text-left p-2">Scalability</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(comparisonResults).map(([algo, info]) => (
                  <tr key={algo} className="border-b">
                    <td className="p-2 font-medium">{algo.toUpperCase()}</td>
                    <td className="p-2">{info.category}</td>
                    <td className="p-2">
                      <span className={`px-2 py-1 text-xs rounded-full ${
                        info.complexity === 'Beginner' ? 'bg-green-100 text-green-800' :
                        info.complexity === 'Intermediate' ? 'bg-yellow-100 text-yellow-800' :
                        info.complexity === 'Advanced' ? 'bg-orange-100 text-orange-800' :
                        'bg-red-100 text-red-800'
                      }`}>
                        {info.complexity}
                      </span>
                    </td>
                    <td className="p-2">{info.sample_efficiency}</td>
                    <td className="p-2">{info.scalability}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">
                🔬 Research Interface
              </h1>
              <p className="text-gray-600">
                Advanced MARL algorithm discovery and experimentation
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="bg-white border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <nav className="flex space-x-8">
            {[
              { id: 'discover', label: '🔍 Discover', desc: 'Algorithm Browser' },
              { id: 'configure', label: '⚙️ Configure', desc: 'Parameter Setup' },
              { id: 'experiment', label: '🧪 Experiment', desc: 'Design Studies' },
              { id: 'compare', label: '📊 Compare', desc: 'Algorithm Analysis' },
              { id: 'optimize', label: '🎯 Optimize', desc: 'Hyperparameter Search' }
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`py-4 px-1 border-b-2 font-medium text-sm ${
                  activeTab === tab.id
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <div>{tab.label}</div>
                <div className="text-xs text-gray-400">{tab.desc}</div>
              </button>
            ))}
          </nav>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <AnimatePresence mode="wait">
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3 }}
          >
            {activeTab === 'discover' && renderAlgorithmBrowser()}
            {activeTab === 'configure' && (
              <div className="space-y-6">
                <div>
                  <h2 className="text-2xl font-bold mb-4">⚙️ Parameter Configuration</h2>
                  <p className="text-gray-600 mb-6">
                    Configure detailed parameters for MARL algorithms with templates and individual agent settings
                  </p>
                </div>

                <div className="bg-white rounded-lg shadow p-6">
                  <div className="grid md:grid-cols-2 gap-6 mb-6">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Select Algorithm
                      </label>
                      <select
                        value={selectedAlgorithm}
                        onChange={(e) => setSelectedAlgorithm(e.target.value)}
                        className="w-full p-3 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
                      >
                        <option value="">Choose an algorithm...</option>
                        {Object.keys(algorithms).map(category => 
                          algorithms[category].map(algo => (
                            <option key={algo} value={algo}>{algo.toUpperCase()}</option>
                          ))
                        )}
                      </select>
                    </div>
                    
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Number of Agents
                      </label>
                      <input
                        type="number"
                        min="1"
                        max="10"
                        value={nAgents}
                        onChange={(e) => setNAgents(parseInt(e.target.value))}
                        className="w-full p-3 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
                      />
                    </div>
                  </div>

                  <div className="grid md:grid-cols-3 gap-4 mb-6">
                    <div className="flex items-center space-x-2">
                      <input
                        type="checkbox"
                        id="memory-shared"
                        checked={memoryShared}
                        onChange={(e) => setMemoryShared(e.target.checked)}
                        className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                      />
                      <label htmlFor="memory-shared" className="text-sm font-medium text-gray-700">
                        Shared Memory
                      </label>
                    </div>
                    
                    <div className="flex items-center space-x-2">
                      <input
                        type="checkbox"
                        id="communication-enabled"
                        checked={communicationEnabled}
                        onChange={(e) => setCommunicationEnabled(e.target.checked)}
                        className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                      />
                      <label htmlFor="communication-enabled" className="text-sm font-medium text-gray-700">
                        Enable Communication
                      </label>
                    </div>
                    
                    <button
                      onClick={() => {
                        if (selectedAlgorithm) {
                          // Load algorithm template
                          fetch(`/api/config/algorithm-template/${selectedAlgorithm}`)
                            .then(res => res.json())
                            .then(data => {
                              if (data.success) {
                                console.log('Algorithm template loaded:', data.template);
                              }
                            });
                        }
                      }}
                      className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
                    >
                      Load Template
                    </button>
                  </div>

                  {selectedAlgorithm && (
                    <div className="border-t pt-6">
                      <ParameterConfigurator
                        algorithm={selectedAlgorithm}
                        onConfigChange={setParameterConfig}
                        initialConfig={parameterConfig}
                      />
                    </div>
                  )}
                </div>

                {selectedAlgorithm && (
                  <div className="bg-white rounded-lg shadow p-6">
                    <h3 className="text-lg font-semibold mb-4">Individual Agent Configuration</h3>
                    <div className="space-y-4">
                      {Array.from({ length: nAgents }, (_, i) => (
                        <div key={i} className="border rounded-lg p-4 bg-gray-50">
                          <h4 className="font-medium text-gray-900 mb-3">Agent {i + 1}</h4>
                          <div className="grid md:grid-cols-3 gap-4">
                            <div>
                              <label className="block text-sm font-medium text-gray-700 mb-1">
                                Learning Rate
                              </label>
                              <input
                                type="number"
                                step="0.0001"
                                min="0.0001"
                                max="0.1"
                                defaultValue="0.001"
                                className="w-full p-2 border border-gray-300 rounded-md text-sm"
                                onChange={(e) => {
                                  const newAgentConfigs = [...agentConfigs];
                                  if (!newAgentConfigs[i]) newAgentConfigs[i] = {};
                                  newAgentConfigs[i].learning_rate = parseFloat(e.target.value);
                                  setAgentConfigs(newAgentConfigs);
                                }}
                              />
                            </div>
                            <div>
                              <label className="block text-sm font-medium text-gray-700 mb-1">
                                Exploration Rate
                              </label>
                              <input
                                type="number"
                                step="0.01"
                                min="0"
                                max="1"
                                defaultValue="0.1"
                                className="w-full p-2 border border-gray-300 rounded-md text-sm"
                                onChange={(e) => {
                                  const newAgentConfigs = [...agentConfigs];
                                  if (!newAgentConfigs[i]) newAgentConfigs[i] = {};
                                  newAgentConfigs[i].epsilon = parseFloat(e.target.value);
                                  setAgentConfigs(newAgentConfigs);
                                }}
                              />
                            </div>
                            <div>
                              <label className="block text-sm font-medium text-gray-700 mb-1">
                                Memory Size
                              </label>
                              <input
                                type="number"
                                step="1000"
                                min="1000"
                                max="100000"
                                defaultValue="10000"
                                className="w-full p-2 border border-gray-300 rounded-md text-sm"
                                onChange={(e) => {
                                  const newAgentConfigs = [...agentConfigs];
                                  if (!newAgentConfigs[i]) newAgentConfigs[i] = {};
                                  newAgentConfigs[i].memory_size = parseInt(e.target.value);
                                  setAgentConfigs(newAgentConfigs);
                                }}
                              />
                            </div>
                          </div>
                          
                          <div className="mt-3 grid md:grid-cols-2 gap-4">
                            <div className="flex items-center space-x-2">
                              <input
                                type="checkbox"
                                id={`agent-${i}-shared-memory`}
                                defaultChecked={memoryShared}
                                className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                                onChange={(e) => {
                                  const newAgentConfigs = [...agentConfigs];
                                  if (!newAgentConfigs[i]) newAgentConfigs[i] = {};
                                  newAgentConfigs[i].shared_memory = e.target.checked;
                                  setAgentConfigs(newAgentConfigs);
                                }}
                              />
                              <label htmlFor={`agent-${i}-shared-memory`} className="text-sm text-gray-700">
                                Use Shared Memory
                              </label>
                            </div>
                            
                            <div className="flex items-center space-x-2">
                              <input
                                type="checkbox"
                                id={`agent-${i}-communication`}
                                defaultChecked={communicationEnabled}
                                className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                                onChange={(e) => {
                                  const newAgentConfigs = [...agentConfigs];
                                  if (!newAgentConfigs[i]) newAgentConfigs[i] = {};
                                  newAgentConfigs[i].communication_enabled = e.target.checked;
                                  setAgentConfigs(newAgentConfigs);
                                }}
                              />
                              <label htmlFor={`agent-${i}-communication`} className="text-sm text-gray-700">
                                Enable Communication
                              </label>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                    
                    <div className="mt-6 flex justify-end space-x-3">
                      <button
                        onClick={() => {
                          // Save configuration
                          const configData = {
                            algorithm: selectedAlgorithm,
                            n_agents: nAgents,
                            custom_params: {
                              ...parameterConfig,
                              memory_shared: memoryShared,
                              communication_enabled: communicationEnabled,
                              agent_configs: agentConfigs
                            }
                          };
                          
                          fetch('/api/config/create-from-template', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify(configData)
                          })
                          .then(res => res.json())
                          .then(data => {
                            if (data.success) {
                              alert('Configuration saved successfully!');
                            } else {
                              alert('Failed to save configuration: ' + data.error);
                            }
                          });
                        }}
                        className="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 transition-colors"
                      >
                        Save Configuration
                      </button>
                      
                      <button
                        onClick={() => {
                          // Start training with current configuration
                          alert('Training with custom configuration not yet implemented');
                        }}
                        className="px-4 py-2 bg-purple-600 text-white rounded-md hover:bg-purple-700 transition-colors"
                      >
                        Start Training
                      </button>
                    </div>
                  </div>
                )}
              </div>
            )}
            {activeTab === 'experiment' && renderExperimentDesigner()}
            {activeTab === 'compare' && renderComparison()}
            {activeTab === 'optimize' && (
              <div className="text-center py-12">
                <h2 className="text-2xl font-bold mb-4">🎯 Hyperparameter Optimization</h2>
                <p className="text-gray-600">
                  Advanced optimization features coming soon...
                </p>
                {optimizationStatus && (
                  <div className="mt-4 p-4 bg-blue-50 rounded-lg">
                    <p>Optimization Status: {optimizationStatus}</p>
                  </div>
                )}
              </div>
            )}
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  );
};

export default ResearchPage;
