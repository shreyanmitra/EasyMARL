import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { PlayIcon, StopIcon, ArrowUpIcon, ArrowDownIcon, ArrowLeftIcon, ArrowRightIcon } from '@heroicons/react/outline';
import toast from 'react-hot-toast';

/**
 * Manual Control Component for MultiGrid Environment
 * 
 * This component allows users to manually control agents in the MultiGrid environment,
 * similar to the manual_control_multigrid.py script but in a web interface.
 * 
 * Features:
 * - Interactive environment visualization
 * - Keyboard and button controls for agent actions
 * - Real-time action feedback
 * - Multi-agent coordination
 * - Environment reset functionality
 */
const ManualControl = () => {
  // Environment state
  const [environment, setEnvironment] = useState(null);
  const [environmentState, setEnvironmentState] = useState(null);
  const [isInitialized, setIsInitialized] = useState(false);

  // Agent state
  const [agents, setAgents] = useState([]);
  const [selectedAgent, setSelectedAgent] = useState(0);
  const [lastActions, setLastActions] = useState([]);

  // Control state
  const [isControlActive, setIsControlActive] = useState(false);
  const [stepCount, setStepCount] = useState(0);
  const [totalReward, setTotalReward] = useState(0);

  // Available actions for MultiGrid (based on manual_control_multigrid.py)
  const actions = [
    { id: 0, name: 'Stay', key: 'Space', icon: '⏸️' },
    { id: 1, name: 'Turn Left', key: 'Q', icon: '↶' },
    { id: 2, name: 'Turn Right', key: 'E', icon: '↷' },
    { id: 3, name: 'Move Forward', key: 'W', icon: '⬆️' },
    { id: 4, name: 'Pick Up', key: 'F', icon: '🤏' },
    { id: 5, name: 'Drop', key: 'R', icon: '🫳' },
    { id: 6, name: 'Toggle', key: 'T', icon: '🔄' },
    { id: 7, name: 'Done', key: 'Enter', icon: '✅' }
  ];

  // Initialize environment
  const initializeEnvironment = async (envName = 'MultiGrid-Cluttered-Fixed-15x15') => {
    try {
      const response = await fetch('/api/manual-control/init', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ environment: envName })
      });

      const data = await response.json();
      
      if (data.success) {
        setEnvironment(data.environment);
        setEnvironmentState(data.state);
        setAgents(data.agents);
        setIsInitialized(true);
        setStepCount(0);
        setTotalReward(0);
        toast.success(`Environment ${envName} initialized with ${data.agents.length} agents`);
      } else {
        toast.error('Failed to initialize environment');
      }
    } catch (error) {
      console.error('Error initializing environment:', error);
      toast.error('Error connecting to backend');
    }
  };

  // Send action to environment
  const sendAction = async (agentId, actionId) => {
    if (!isInitialized || !isControlActive) return;

    try {
      const response = await fetch('/api/manual-control/step', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          agent_id: agentId,
          action: actionId
        })
      });

      const data = await response.json();
      
      if (data.success) {
        setEnvironmentState(data.state);
        setStepCount(data.step_count);
        setTotalReward(data.total_reward);
        setLastActions(data.actions);
        
        // Show action feedback
        const actionName = actions.find(a => a.id === actionId)?.name || 'Unknown';
        toast.success(`Agent ${agentId}: ${actionName}`);
        
        // Check if episode is done
        if (data.done) {
          setIsControlActive(false);
          toast.success(`Episode complete! Total reward: ${data.total_reward}`);
        }
      } else {
        toast.error('Failed to execute action');
      }
    } catch (error) {
      console.error('Error sending action:', error);
      toast.error('Error communicating with environment');
    }
  };

  // Reset environment
  const resetEnvironment = async () => {
    try {
      const response = await fetch('/api/manual-control/reset', {
        method: 'POST'
      });

      const data = await response.json();
      
      if (data.success) {
        setEnvironmentState(data.state);
        setStepCount(0);
        setTotalReward(0);
        setLastActions([]);
        toast.success('Environment reset');
      }
    } catch (error) {
      console.error('Error resetting environment:', error);
      toast.error('Failed to reset environment');
    }
  };

  // Keyboard event handler
  useEffect(() => {
    const handleKeyPress = (event) => {
      if (!isControlActive) return;

      const key = event.key.toLowerCase();
      let actionId = null;

      // Map keyboard keys to actions
      switch (key) {
        case ' ':
          actionId = 0; // Stay
          break;
        case 'q':
          actionId = 1; // Turn Left
          break;
        case 'e':
          actionId = 2; // Turn Right
          break;
        case 'w':
          actionId = 3; // Move Forward
          break;
        case 'f':
          actionId = 4; // Pick Up
          break;
        case 'r':
          actionId = 5; // Drop
          break;
        case 't':
          actionId = 6; // Toggle
          break;
        case 'enter':
          actionId = 7; // Done
          break;
        default:
          return;
      }

      if (actionId !== null) {
        event.preventDefault();
        sendAction(selectedAgent, actionId);
      }
    };

    if (isControlActive) {
      window.addEventListener('keydown', handleKeyPress);
    }

    return () => {
      window.removeEventListener('keydown', handleKeyPress);
    };
  }, [isControlActive, selectedAgent]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-8"
        >
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            🎮 Manual Environment Control
          </h1>
          <p className="text-gray-600 max-w-2xl mx-auto">
            Take direct control of agents in the MultiGrid environment. 
            Use keyboard controls or click buttons to guide your agents through the environment.
          </p>
        </motion.div>

        {/* Control Panel */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
          {/* Environment Setup */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="bg-white rounded-xl shadow-lg p-6"
          >
            <h3 className="text-lg font-semibold mb-4">🌍 Environment Setup</h3>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Environment
                </label>
                <select 
                  className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                  defaultValue="MultiGrid-Cluttered-Fixed-15x15"
                >
                  <option value="MultiGrid-Cluttered-Fixed-15x15">Cluttered 15x15</option>
                  <option value="MultiGrid-DoorKey-8x8-v0">Door Key 8x8</option>
                  <option value="MultiGrid-Empty-8x8-v0">Empty 8x8</option>
                  <option value="MultiGrid-FourRooms-v0">Four Rooms</option>
                </select>
              </div>

              <button
                onClick={() => initializeEnvironment()}
                disabled={isControlActive}
                className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white px-4 py-2 rounded-lg flex items-center justify-center gap-2"
              >
                <PlayIcon className="w-5 h-5" />
                Initialize Environment
              </button>

              {isInitialized && (
                <>
                  <button
                    onClick={() => setIsControlActive(!isControlActive)}
                    className={`w-full px-4 py-2 rounded-lg flex items-center justify-center gap-2 ${
                      isControlActive 
                        ? 'bg-red-600 hover:bg-red-700 text-white' 
                        : 'bg-green-600 hover:bg-green-700 text-white'
                    }`}
                  >
                    <StopIcon className="w-5 h-5" />
                    {isControlActive ? 'Stop Control' : 'Start Control'}
                  </button>

                  <button
                    onClick={resetEnvironment}
                    className="w-full bg-gray-600 hover:bg-gray-700 text-white px-4 py-2 rounded-lg"
                  >
                    🔄 Reset Environment
                  </button>
                </>
              )}
            </div>
          </motion.div>

          {/* Agent Selection */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-white rounded-xl shadow-lg p-6"
          >
            <h3 className="text-lg font-semibold mb-4">🤖 Agent Control</h3>
            
            {agents.length > 0 ? (
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Selected Agent
                  </label>
                  <select 
                    value={selectedAgent}
                    onChange={(e) => setSelectedAgent(Number(e.target.value))}
                    className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                  >
                    {agents.map((agent, index) => (
                      <option key={index} value={index}>
                        Agent {index} {agent.position ? `(${agent.position.x}, ${agent.position.y})` : ''}
                      </option>
                    ))}
                  </select>
                </div>

                <div className="text-sm text-gray-600">
                  <p>Steps: {stepCount}</p>
                  <p>Total Reward: {totalReward.toFixed(2)}</p>
                  <p>Status: {isControlActive ? '🟢 Active' : '🔴 Inactive'}</p>
                </div>
              </div>
            ) : (
              <p className="text-gray-500 text-center py-8">
                Initialize environment to see agents
              </p>
            )}
          </motion.div>

          {/* Action Controls */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="bg-white rounded-xl shadow-lg p-6"
          >
            <h3 className="text-lg font-semibold mb-4">🎯 Action Controls</h3>
            
            {isControlActive ? (
              <div className="grid grid-cols-2 gap-2">
                {actions.map((action) => (
                  <button
                    key={action.id}
                    onClick={() => sendAction(selectedAgent, action.id)}
                    className="bg-gray-100 hover:bg-gray-200 border border-gray-300 rounded-lg p-3 text-center transition-colors"
                    title={`Press ${action.key}`}
                  >
                    <div className="text-lg mb-1">{action.icon}</div>
                    <div className="text-xs font-medium">{action.name}</div>
                    <div className="text-xs text-gray-500">{action.key}</div>
                  </button>
                ))}
              </div>
            ) : (
              <p className="text-gray-500 text-center py-8">
                Start control to enable actions
              </p>
            )}

            {isControlActive && (
              <div className="mt-4 p-3 bg-blue-50 rounded-lg">
                <p className="text-sm text-blue-800 font-medium">💡 Tip:</p>
                <p className="text-xs text-blue-700">
                  Use keyboard shortcuts or click buttons to control Agent {selectedAgent}
                </p>
              </div>
            )}
          </motion.div>
        </div>

        {/* Environment Visualization */}
        {isInitialized && environmentState && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-white rounded-xl shadow-lg p-6"
          >
            <h3 className="text-lg font-semibold mb-4">🗺️ Environment View</h3>
            
            <div className="flex justify-center">
              <div className="border border-gray-300 rounded-lg p-4 bg-gray-50">
                {/* Environment grid visualization */}
                <div style={{ 
                  display: 'grid', 
                  gridTemplateColumns: `repeat(${environmentState.width || 15}, 30px)`,
                  gap: '1px',
                  backgroundColor: '#e5e7eb'
                }}>
                  {environmentState.grid?.map((cell, index) => (
                    <div
                      key={index}
                      className="w-8 h-8 flex items-center justify-center text-sm font-bold"
                      style={{ backgroundColor: cell.color || '#ffffff' }}
                      title={`${cell.type} at (${index % (environmentState.width || 15)}, ${Math.floor(index / (environmentState.width || 15))})`}
                    >
                      {cell.emoji || cell.char || ''}
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {lastActions.length > 0 && (
              <div className="mt-4 p-3 bg-green-50 rounded-lg">
                <p className="text-sm font-medium text-green-800">Last Actions:</p>
                <p className="text-sm text-green-700">
                  {lastActions.map((action, index) => 
                    `Agent ${index}: ${actions.find(a => a.id === action)?.name || action}`
                  ).join(', ')}
                </p>
              </div>
            )}
          </motion.div>
        )}

        {/* Instructions */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white rounded-xl shadow-lg p-6 mt-6"
        >
          <h3 className="text-lg font-semibold mb-4">📖 Instructions</h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-medium text-gray-900 mb-2">🎮 Controls</h4>
              <ul className="text-sm text-gray-700 space-y-1">
                <li><kbd className="px-1 py-0.5 bg-gray-200 rounded">Space</kbd> - Stay in place</li>
                <li><kbd className="px-1 py-0.5 bg-gray-200 rounded">Q</kbd> - Turn left</li>
                <li><kbd className="px-1 py-0.5 bg-gray-200 rounded">E</kbd> - Turn right</li>
                <li><kbd className="px-1 py-0.5 bg-gray-200 rounded">W</kbd> - Move forward</li>
                <li><kbd className="px-1 py-0.5 bg-gray-200 rounded">F</kbd> - Pick up object</li>
                <li><kbd className="px-1 py-0.5 bg-gray-200 rounded">R</kbd> - Drop object</li>
                <li><kbd className="px-1 py-0.5 bg-gray-200 rounded">T</kbd> - Toggle/interact</li>
                <li><kbd className="px-1 py-0.5 bg-gray-200 rounded">Enter</kbd> - Mark done</li>
              </ul>
            </div>
            <div>
              <h4 className="font-medium text-gray-900 mb-2">🎯 How to Use</h4>
              <ol className="text-sm text-gray-700 space-y-1 list-decimal list-inside">
                <li>Initialize an environment from the dropdown</li>
                <li>Click "Start Control" to enable manual control</li>
                <li>Select which agent to control</li>
                <li>Use keyboard keys or click action buttons</li>
                <li>Watch agents move in real-time</li>
                <li>Reset anytime to start over</li>
              </ol>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
};

export default ManualControl;
