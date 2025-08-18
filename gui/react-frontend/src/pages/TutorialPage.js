import React, { useState } from 'react';

/**
 * Comprehensive Tutorial Page for EasyMARL
 * 
 * Interactive learning guide that takes users from MARL basics
 * to advanced algorithm implementation and research applications.
 */
const TutorialPage = () => {
  const [currentSection, setCurrentSection] = useState('basics');
  const [completedSections, setCompletedSections] = useState(new Set());

  const markCompleted = (section) => {
    setCompletedSections(prev => new Set([...prev, section]));
  };

  const tutorials = {
    basics: {
      title: "🎯 MARL Basics",
      duration: "15 min",
      description: "Understanding Multi-Agent Reinforcement Learning fundamentals",
      content: (
        <div className="space-y-6">
          <div className="bg-blue-50 p-6 rounded-lg">
            <h3 className="text-xl font-bold mb-4">What is Multi-Agent Reinforcement Learning?</h3>
            <p className="mb-4">
              MARL extends single-agent RL to environments with multiple learning agents.
              Think of it as teaching multiple AI agents to work together or compete.
            </p>
            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-white p-4 rounded">
                <h4 className="font-semibold text-green-600">🤝 Cooperative</h4>
                <p className="text-sm">Agents work toward common goals (like a sports team)</p>
                <ul className="text-xs mt-2 list-disc list-inside">
                  <li>Shared rewards</li>
                  <li>Common objectives</li>
                  <li>Team coordination</li>
                </ul>
              </div>
              <div className="bg-white p-4 rounded">
                <h4 className="font-semibold text-red-600">⚔️ Competitive</h4>
                <p className="text-sm">Agents compete against each other (like chess)</p>
                <ul className="text-xs mt-2 list-disc list-inside">
                  <li>Individual rewards</li>
                  <li>Strategic play</li>
                  <li>Game theory</li>
                </ul>
              </div>
            </div>
          </div>

          <div className="bg-gray-50 p-6 rounded-lg">
            <h3 className="text-lg font-bold mb-3">Key Challenges in MARL</h3>
            <div className="grid md:grid-cols-3 gap-4">
              <div className="bg-white p-3 rounded">
                <h4 className="font-semibold text-orange-600">🎯 Credit Assignment</h4>
                <p className="text-sm">Which agent contributed to success/failure?</p>
              </div>
              <div className="bg-white p-3 rounded">
                <h4 className="font-semibold text-purple-600">🌊 Non-Stationarity</h4>
                <p className="text-sm">Environment changes as other agents learn</p>
              </div>
              <div className="bg-white p-3 rounded">
                <h4 className="font-semibold text-blue-600">📈 Scalability</h4>
                <p className="text-sm">How to handle many agents efficiently?</p>
              </div>
            </div>
          </div>

          <div className="bg-green-50 p-6 rounded-lg">
            <h3 className="text-lg font-bold mb-3">Your Learning Path</h3>
            <div className="space-y-2">
              <div className="flex items-center">
                <span className="bg-green-500 text-white px-2 py-1 rounded text-xs mr-3">1</span>
                <span>Start with simple algorithms (IPPO, IQL)</span>
              </div>
              <div className="flex items-center">
                <span className="bg-blue-500 text-white px-2 py-1 rounded text-xs mr-3">2</span>
                <span>Learn coordination methods (QMIX, VDN)</span>
              </div>
              <div className="flex items-center">
                <span className="bg-purple-500 text-white px-2 py-1 rounded text-xs mr-3">3</span>
                <span>Advanced techniques (MAPPO, MADDPG)</span>
              </div>
              <div className="flex items-center">
                <span className="bg-red-500 text-white px-2 py-1 rounded text-xs mr-3">4</span>
                <span>Specialized methods (COMA, NFSP)</span>
              </div>
            </div>
          </div>
        </div>
      )
    },

    algorithms: {
      title: "🧠 Algorithm Guide",
      duration: "25 min", 
      description: "Understanding different MARL algorithms and when to use them",
      content: (
        <div className="space-y-6">
          <div className="bg-yellow-50 p-6 rounded-lg">
            <h3 className="text-xl font-bold mb-4">Algorithm Categories</h3>
            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <h4 className="font-bold text-blue-600 mb-3">🎯 Value-Based Methods</h4>
                <div className="space-y-3">
                  <div className="bg-white p-3 rounded border-l-4 border-blue-400">
                    <h5 className="font-semibold">IQL (Independent Q-Learning)</h5>
                    <p className="text-sm text-gray-600">Perfect for beginners - just Q-learning per agent</p>
                    <div className="mt-2">
                      <span className="bg-green-100 text-green-800 px-2 py-1 rounded text-xs">Beginner</span>
                      <span className="bg-blue-100 text-blue-800 px-2 py-1 rounded text-xs ml-1">Simple</span>
                    </div>
                  </div>
                  <div className="bg-white p-3 rounded border-l-4 border-blue-400">
                    <h5 className="font-semibold">QMIX</h5>
                    <p className="text-sm text-gray-600">Learns to mix individual Q-values for cooperation</p>
                    <div className="mt-2">
                      <span className="bg-yellow-100 text-yellow-800 px-2 py-1 rounded text-xs">Intermediate</span>
                      <span className="bg-purple-100 text-purple-800 px-2 py-1 rounded text-xs ml-1">Coordination</span>
                    </div>
                  </div>
                  <div className="bg-white p-3 rounded border-l-4 border-blue-400">
                    <h5 className="font-semibold">VDN</h5>
                    <p className="text-sm text-gray-600">Simple additive value decomposition</p>
                    <div className="mt-2">
                      <span className="bg-green-100 text-green-800 px-2 py-1 rounded text-xs">Beginner</span>
                      <span className="bg-gray-100 text-gray-800 px-2 py-1 rounded text-xs ml-1">Foundation</span>
                    </div>
                  </div>
                </div>
              </div>

              <div>
                <h4 className="font-bold text-green-600 mb-3">🎭 Policy-Based Methods</h4>
                <div className="space-y-3">
                  <div className="bg-white p-3 rounded border-l-4 border-green-400">
                    <h5 className="font-semibold">IPPO (Independent PPO)</h5>
                    <p className="text-sm text-gray-600">PPO applied independently to each agent</p>
                    <div className="mt-2">
                      <span className="bg-green-100 text-green-800 px-2 py-1 rounded text-xs">Beginner</span>
                      <span className="bg-blue-100 text-blue-800 px-2 py-1 rounded text-xs ml-1">Stable</span>
                    </div>
                  </div>
                  <div className="bg-white p-3 rounded border-l-4 border-green-400">
                    <h5 className="font-semibold">MAPPO</h5>
                    <p className="text-sm text-gray-600">State-of-the-art with centralized training</p>
                    <div className="mt-2">
                      <span className="bg-red-100 text-red-800 px-2 py-1 rounded text-xs">Advanced</span>
                      <span className="bg-purple-100 text-purple-800 px-2 py-1 rounded text-xs ml-1">SOTA</span>
                    </div>
                  </div>
                  <div className="bg-white p-3 rounded border-l-4 border-green-400">
                    <h5 className="font-semibold">MADDPG</h5>
                    <p className="text-sm text-gray-600">Centralized training, decentralized execution</p>
                    <div className="mt-2">
                      <span className="bg-yellow-100 text-yellow-800 px-2 py-1 rounded text-xs">Intermediate</span>
                      <span className="bg-orange-100 text-orange-800 px-2 py-1 rounded text-xs ml-1">Continuous</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-purple-50 p-6 rounded-lg">
            <h3 className="text-lg font-bold mb-3">🎯 Algorithm Selection Guide</h3>
            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-white p-4 rounded">
                <h4 className="font-semibold text-green-600 mb-2">First Time with MARL?</h4>
                <p className="text-sm mb-2">Start here to build intuition:</p>
                <ul className="text-sm space-y-1">
                  <li>1️⃣ <strong>IQL</strong> - Understand the basics</li>
                  <li>2️⃣ <strong>IPPO</strong> - Learn policy gradients</li>
                  <li>3️⃣ <strong>VDN</strong> - See coordination in action</li>
                </ul>
              </div>
              <div className="bg-white p-4 rounded">
                <h4 className="font-semibold text-blue-600 mb-2">Need Good Performance?</h4>
                <p className="text-sm mb-2">Production-ready algorithms:</p>
                <ul className="text-sm space-y-1">
                  <li>🏆 <strong>MAPPO</strong> - Best overall performance</li>
                  <li>🎯 <strong>QMIX</strong> - Great for discrete actions</li>
                  <li>⚡ <strong>MADDPG</strong> - Continuous control</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      )
    },

    handson: {
      title: "🛠️ Hands-On Training",
      duration: "30 min",
      description: "Step-by-step guide to training your first MARL agents",
      content: (
        <div className="space-y-6">
          <div className="bg-blue-50 p-6 rounded-lg">
            <h3 className="text-xl font-bold mb-4">Your First MARL Experiment</h3>
            <div className="bg-white p-4 rounded mb-4">
              <h4 className="font-semibold mb-2">🎯 Goal</h4>
              <p>Train 2 agents to cooperatively navigate a cluttered environment and reach their goals.</p>
            </div>
            
            <div className="space-y-4">
              <div className="border-l-4 border-blue-400 pl-4">
                <h4 className="font-semibold">Step 1: Choose Your Algorithm</h4>
                <p className="text-sm text-gray-600">We recommend starting with <strong>IPPO</strong> for your first experiment.</p>
                <div className="mt-2 p-2 bg-gray-100 rounded">
                  <p className="text-xs"><strong>Why IPPO?</strong> It's simple, stable, and gives good results for cooperative tasks.</p>
                </div>
              </div>

              <div className="border-l-4 border-green-400 pl-4">
                <h4 className="font-semibold">Step 2: Configure Training</h4>
                <div className="mt-2 bg-gray-100 p-3 rounded">
                  <p className="text-sm font-semibold mb-2">Recommended Settings:</p>
                  <ul className="text-xs space-y-1">
                    <li>• <strong>Environment:</strong> MultiGrid-Cluttered-Fixed-15x15</li>
                    <li>• <strong>Episodes:</strong> 500 (about 10 minutes)</li>
                    <li>• <strong>Learning Rate:</strong> 0.001 (default)</li>
                    <li>• <strong>Enable WandB:</strong> Yes (for tracking)</li>
                  </ul>
                </div>
              </div>

              <div className="border-l-4 border-purple-400 pl-4">
                <h4 className="font-semibold">Step 3: Monitor Training</h4>
                <p className="text-sm text-gray-600">Watch the real-time charts to see your agents learning!</p>
                <div className="mt-2 grid grid-cols-2 gap-2">
                  <div className="bg-white p-2 rounded border">
                    <p className="text-xs font-semibold">📈 Episode Rewards</p>
                    <p className="text-xs text-gray-600">Should increase over time</p>
                  </div>
                  <div className="bg-white p-2 rounded border">
                    <p className="text-xs font-semibold">📊 Episode Length</p>
                    <p className="text-xs text-gray-600">May decrease as agents get smarter</p>
                  </div>
                </div>
              </div>

              <div className="border-l-4 border-red-400 pl-4">
                <h4 className="font-semibold">Step 4: Analyze Results</h4>
                <p className="text-sm text-gray-600">After training, evaluate your agents' performance.</p>
                <div className="mt-2 p-2 bg-yellow-100 rounded">
                  <p className="text-xs"><strong>Success indicators:</strong> Increasing rewards, agents reaching goals, coordinated movement</p>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-green-50 p-6 rounded-lg">
            <h3 className="text-lg font-bold mb-3">🚀 Try These Experiments</h3>
            <div className="space-y-3">
              <div className="bg-white p-3 rounded">
                <h4 className="font-semibold text-blue-600">Experiment 1: Algorithm Comparison</h4>
                <p className="text-sm">Train IPPO, QMIX, and MAPPO on the same environment. Which performs best?</p>
              </div>
              <div className="bg-white p-3 rounded">
                <h4 className="font-semibold text-green-600">Experiment 2: Environment Difficulty</h4>
                <p className="text-sm">Try different environments: Empty → Cluttered → Maze. How does performance change?</p>
              </div>
              <div className="bg-white p-3 rounded">
                <h4 className="font-semibold text-purple-600">Experiment 3: Hyperparameter Tuning</h4>
                <p className="text-sm">Adjust learning rates and see how it affects training speed and stability.</p>
              </div>
            </div>
          </div>
        </div>
      )
    },

    research: {
      title: "🔬 Research Applications",
      duration: "20 min",
      description: "Using EasyMARL for research and advanced applications",
      content: (
        <div className="space-y-6">
          <div className="bg-purple-50 p-6 rounded-lg">
            <h3 className="text-xl font-bold mb-4">Research with EasyMARL</h3>
            <p className="mb-4">
              EasyMARL is designed for both education and cutting-edge research. Here's how to leverage
              it for your research projects.
            </p>
            
            <div className="grid md:grid-cols-2 gap-6">
              <div className="bg-white p-4 rounded">
                <h4 className="font-bold text-blue-600 mb-3">🔬 Research Areas</h4>
                <ul className="space-y-2 text-sm">
                  <li>• <strong>Algorithm Development:</strong> Implement new MARL algorithms</li>
                  <li>• <strong>Coordination:</strong> Study emergent team behaviors</li>
                  <li>• <strong>Communication:</strong> Design agent communication protocols</li>
                  <li>• <strong>Scalability:</strong> Test with varying numbers of agents</li>
                  <li>• <strong>Transfer Learning:</strong> Apply learned policies to new environments</li>
                </ul>
              </div>
              
              <div className="bg-white p-4 rounded">
                <h4 className="font-bold text-green-600 mb-3">📊 Data Collection</h4>
                <ul className="space-y-2 text-sm">
                  <li>• <strong>WandB Integration:</strong> Professional experiment tracking</li>
                  <li>• <strong>Comprehensive Metrics:</strong> Rewards, lengths, success rates</li>
                  <li>• <strong>Model Checkpoints:</strong> Save and restore training states</li>
                  <li>• <strong>Visualization:</strong> Generate videos and plots</li>
                  <li>• <strong>CSV Export:</strong> Export data for further analysis</li>
                </ul>
              </div>
            </div>
          </div>

          <div className="bg-orange-50 p-6 rounded-lg">
            <h3 className="text-lg font-bold mb-3">🛠️ Extending the Framework</h3>
            <div className="space-y-4">
              <div className="bg-white p-4 rounded">
                <h4 className="font-semibold">Adding New Algorithms</h4>
                <p className="text-sm text-gray-600 mb-2">Follow the base class structure:</p>
                <div className="bg-gray-100 p-2 rounded text-xs font-mono">
                  <p>class MyAlgorithm(MARLAlgorithm):</p>
                  <p className="ml-4">def __init__(self, env, config, device):</p>
                  <p className="ml-8"># Initialize your algorithm</p>
                  <p className="ml-4">def collect_rollout(self, env):</p>
                  <p className="ml-8"># Collect experience</p>
                  <p className="ml-4">def train_step(self, rollout_data):</p>
                  <p className="ml-8"># Update networks</p>
                </div>
              </div>

              <div className="bg-white p-4 rounded">
                <h4 className="font-semibold">Custom Environments</h4>
                <p className="text-sm text-gray-600">Add your own multi-agent environments following the gym interface.</p>
              </div>

              <div className="bg-white p-4 rounded">
                <h4 className="font-semibold">Advanced Metrics</h4>
                <p className="text-sm text-gray-600">Implement custom evaluation metrics for your specific research questions.</p>
              </div>
            </div>
          </div>

          <div className="bg-red-50 p-6 rounded-lg">
            <h3 className="text-lg font-bold mb-3">📚 Publication Tips</h3>
            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-white p-3 rounded">
                <h4 className="font-semibold text-blue-600">Reproducibility</h4>
                <ul className="text-sm mt-2 space-y-1">
                  <li>• Use fixed seeds for all experiments</li>
                  <li>• Document all hyperparameters</li>
                  <li>• Save configuration files</li>
                  <li>• Include environment details</li>
                </ul>
              </div>
              <div className="bg-white p-3 rounded">
                <h4 className="font-semibold text-green-600">Evaluation</h4>
                <ul className="text-sm mt-2 space-y-1">
                  <li>• Run multiple seeds (at least 5)</li>
                  <li>• Report confidence intervals</li>
                  <li>• Compare against strong baselines</li>
                  <li>• Include statistical significance tests</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      )
    },

    troubleshooting: {
      title: "🐛 Troubleshooting",
      duration: "10 min",
      description: "Common issues and how to solve them",
      content: (
        <div className="space-y-6">
          <div className="bg-red-50 p-6 rounded-lg">
            <h3 className="text-xl font-bold mb-4">Common Issues & Solutions</h3>
            
            <div className="space-y-4">
              <div className="bg-white p-4 rounded border-l-4 border-red-400">
                <h4 className="font-semibold text-red-600">Training Not Starting</h4>
                <div className="mt-2 space-y-2">
                  <p className="text-sm"><strong>Symptoms:</strong> "Start Training" button doesn't work</p>
                  <p className="text-sm"><strong>Solutions:</strong></p>
                  <ul className="text-sm list-disc list-inside ml-4">
                    <li>Check that Flask backend is running</li>
                    <li>Verify environment name is correct</li>
                    <li>Ensure no other training session is active</li>
                  </ul>
                </div>
              </div>

              <div className="bg-white p-4 rounded border-l-4 border-yellow-400">
                <h4 className="font-semibold text-yellow-600">Poor Training Performance</h4>
                <div className="mt-2 space-y-2">
                  <p className="text-sm"><strong>Symptoms:</strong> Rewards not improving, agents not learning</p>
                  <p className="text-sm"><strong>Solutions:</strong></p>
                  <ul className="text-sm list-disc list-inside ml-4">
                    <li>Try different learning rates (0.001, 0.0005, 0.0001)</li>
                    <li>Increase training episodes</li>
                    <li>Switch to a different algorithm (MAPPO often works well)</li>
                    <li>Check environment difficulty - start with easier tasks</li>
                  </ul>
                </div>
              </div>

              <div className="bg-white p-4 rounded border-l-4 border-blue-400">
                <h4 className="font-semibold text-blue-600">Memory/Performance Issues</h4>
                <div className="mt-2 space-y-2">
                  <p className="text-sm"><strong>Symptoms:</strong> Slow training, out of memory errors</p>
                  <p className="text-sm"><strong>Solutions:</strong></p>
                  <ul className="text-sm list-disc list-inside ml-4">
                    <li>Reduce batch size in algorithm configuration</li>
                    <li>Use CPU instead of GPU for small experiments</li>
                    <li>Decrease environment complexity</li>
                    <li>Close browser tabs and restart if needed</li>
                  </ul>
                </div>
              </div>

              <div className="bg-white p-4 rounded border-l-4 border-green-400">
                <h4 className="font-semibold text-green-600">Charts Not Updating</h4>
                <div className="mt-2 space-y-2">
                  <p className="text-sm"><strong>Symptoms:</strong> Training graphs frozen or not showing data</p>
                  <p className="text-sm"><strong>Solutions:</strong></p>
                  <ul className="text-sm list-disc list-inside ml-4">
                    <li>Refresh the page</li>
                    <li>Check browser console for errors</li>
                    <li>Ensure stable internet connection</li>
                    <li>Wait a few seconds - updates every 2 seconds</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-blue-50 p-6 rounded-lg">
            <h3 className="text-lg font-bold mb-3">🎯 Best Practices</h3>
            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-white p-3 rounded">
                <h4 className="font-semibold text-green-600">For Beginners</h4>
                <ul className="text-sm mt-2 space-y-1">
                  <li>• Start with simple environments</li>
                  <li>• Use default hyperparameters first</li>
                  <li>• Train for at least 500 episodes</li>
                  <li>• Enable WandB for better tracking</li>
                </ul>
              </div>
              <div className="bg-white p-3 rounded">
                <h4 className="font-semibold text-blue-600">For Research</h4>
                <ul className="text-sm mt-2 space-y-1">
                  <li>• Always use multiple random seeds</li>
                  <li>• Document all configuration changes</li>
                  <li>• Save model checkpoints regularly</li>
                  <li>• Compare against multiple baselines</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      )
    }
  };

  return (
    <div className="max-w-6xl mx-auto p-6">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">
          🎓 EasyMARL Learning Center
        </h1>
        <p className="text-lg text-gray-600">
          Master Multi-Agent Reinforcement Learning step by step
        </p>
      </div>

      {/* Progress Bar */}
      <div className="mb-8">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm text-gray-600">Learning Progress</span>
          <span className="text-sm text-gray-600">
            {completedSections.size} / {Object.keys(tutorials).length} completed
          </span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-2">
          <div 
            className="bg-blue-500 h-2 rounded-full transition-all duration-300"
            style={{ width: `${(completedSections.size / Object.keys(tutorials).length) * 100}%` }}
          ></div>
        </div>
      </div>

      <div className="grid lg:grid-cols-4 gap-6">
        {/* Navigation Sidebar */}
        <div className="lg:col-span-1">
          <div className="bg-white rounded-lg shadow-sm border p-4 sticky top-6">
            <h2 className="font-bold text-gray-900 mb-4">Tutorial Sections</h2>
            <nav className="space-y-2">
              {Object.entries(tutorials).map(([key, tutorial]) => (
                <button
                  key={key}
                  onClick={() => setCurrentSection(key)}
                  className={`w-full text-left p-3 rounded-lg transition-colors ${
                    currentSection === key
                      ? 'bg-blue-100 text-blue-700 border border-blue-200'
                      : 'hover:bg-gray-50'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="font-medium">{tutorial.title}</div>
                      <div className="text-xs text-gray-500">{tutorial.duration}</div>
                    </div>
                    {completedSections.has(key) && (
                      <span className="text-green-500">✓</span>
                    )}
                  </div>
                </button>
              ))}
            </nav>
          </div>
        </div>

        {/* Tutorial Content */}
        <div className="lg:col-span-3">
          <div className="bg-white rounded-lg shadow-sm border">
            <div className="p-6 border-b">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-2xl font-bold text-gray-900">
                    {tutorials[currentSection].title}
                  </h2>
                  <p className="text-gray-600 mt-1">
                    {tutorials[currentSection].description}
                  </p>
                  <div className="flex items-center mt-2 text-sm text-gray-500">
                    <span>⏱️ {tutorials[currentSection].duration}</span>
                  </div>
                </div>
                <button
                  onClick={() => markCompleted(currentSection)}
                  disabled={completedSections.has(currentSection)}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                    completedSections.has(currentSection)
                      ? 'bg-green-100 text-green-700 cursor-not-allowed'
                      : 'bg-blue-100 text-blue-700 hover:bg-blue-200'
                  }`}
                >
                  {completedSections.has(currentSection) ? '✓ Completed' : 'Mark Complete'}
                </button>
              </div>
            </div>
            
            <div className="p-6">
              {tutorials[currentSection].content}
            </div>

            <div className="p-6 border-t bg-gray-50">
              <div className="flex justify-between">
                <button
                  onClick={() => {
                    const keys = Object.keys(tutorials);
                    const currentIndex = keys.indexOf(currentSection);
                    if (currentIndex > 0) {
                      setCurrentSection(keys[currentIndex - 1]);
                    }
                  }}
                  disabled={Object.keys(tutorials).indexOf(currentSection) === 0}
                  className="px-4 py-2 bg-gray-100 text-gray-600 rounded-lg disabled:opacity-50"
                >
                  ← Previous
                </button>
                <button
                  onClick={() => {
                    const keys = Object.keys(tutorials);
                    const currentIndex = keys.indexOf(currentSection);
                    if (currentIndex < keys.length - 1) {
                      setCurrentSection(keys[currentIndex + 1]);
                    }
                  }}
                  disabled={Object.keys(tutorials).indexOf(currentSection) === Object.keys(tutorials).length - 1}
                  className="px-4 py-2 bg-blue-100 text-blue-700 rounded-lg disabled:opacity-50"
                >
                  Next →
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TutorialPage;
