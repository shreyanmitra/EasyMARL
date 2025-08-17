import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { EyeIcon, PlayIcon, StopIcon, RefreshIcon } from '@heroicons/react/outline';

const EnvironmentVisualizer = ({ trainingData, isTraining }) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [environmentGrid, setEnvironmentGrid] = useState(null);

  // Simulate environment visualization
  useEffect(() => {
    if (trainingData && trainingData.episodes && trainingData.episodes.length > 0) {
      // Create a simple grid visualization
      const gridSize = 10;
      const grid = Array(gridSize).fill().map(() => Array(gridSize).fill('empty'));
      
      // Add agents and goals based on training progress
      const progress = trainingData.episodes.length / 1000; // Assuming max 1000 episodes
      const numAgents = Math.min(4, Math.floor(progress * 4) + 1);
      
      for (let i = 0; i < numAgents; i++) {
        const x = Math.floor(Math.random() * gridSize);
        const y = Math.floor(Math.random() * gridSize);
        grid[x][y] = `agent-${i}`;
      }
      
      // Add goals
      for (let i = 0; i < 2; i++) {
        const x = Math.floor(Math.random() * gridSize);
        const y = Math.floor(Math.random() * gridSize);
        if (grid[x][y] === 'empty') {
          grid[x][y] = 'goal';
        }
      }
      
      setEnvironmentGrid(grid);
    }
  }, [trainingData]);

  // Auto-play simulation
  useEffect(() => {
    if (isPlaying && environmentGrid) {
      const interval = setInterval(() => {
        setCurrentStep(prev => (prev + 1) % 100);
      }, 200);
      return () => clearInterval(interval);
    }
  }, [isPlaying, environmentGrid]);

  const getCellColor = (cellType) => {
    switch (cellType) {
      case 'empty': return 'bg-gray-100';
      case 'goal': return 'bg-green-400';
      case 'agent-0': return 'bg-blue-500';
      case 'agent-1': return 'bg-red-500';
      case 'agent-2': return 'bg-yellow-500';
      case 'agent-3': return 'bg-purple-500';
      default: return 'bg-gray-100';
    }
  };

  const getCellIcon = (cellType) => {
    if (cellType.startsWith('agent-')) {
      return '🤖';
    } else if (cellType === 'goal') {
      return '🎯';
    }
    return '';
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-white rounded-lg shadow-lg p-6"
    >
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-900 flex items-center">
          <EyeIcon className="h-5 w-5 mr-2 text-blue-600" />
          Environment Visualization
        </h3>
        
        <div className="flex space-x-2">
          <button
            onClick={() => setIsPlaying(!isPlaying)}
            className={`px-3 py-1 rounded-md text-sm font-medium transition-colors ${
              isPlaying 
                ? 'bg-red-100 text-red-700 hover:bg-red-200' 
                : 'bg-green-100 text-green-700 hover:bg-green-200'
            }`}
          >
            {isPlaying ? (
              <><StopIcon className="h-4 w-4 inline mr-1" />Pause</>
            ) : (
              <><PlayIcon className="h-4 w-4 inline mr-1" />Play</>
            )}
          </button>
          
          <button
            onClick={() => setCurrentStep(0)}
            className="px-3 py-1 rounded-md text-sm font-medium bg-gray-100 text-gray-700 hover:bg-gray-200 transition-colors"
          >
            <RefreshIcon className="h-4 w-4 inline mr-1" />Reset
          </button>
        </div>
      </div>

      {/* Environment Grid */}
      {environmentGrid ? (
        <div className="space-y-4">
          <div className="grid grid-cols-10 gap-1 p-4 bg-gray-50 rounded-lg">
            {environmentGrid.flat().map((cell, index) => (
              <motion.div
                key={index}
                className={`aspect-square rounded-sm border border-gray-200 flex items-center justify-center text-xs ${getCellColor(cell)}`}
                whileHover={{ scale: 1.1 }}
                animate={{
                  scale: cell.startsWith('agent-') ? 1 + 0.1 * Math.sin(currentStep * 0.2) : 1
                }}
              >
                {getCellIcon(cell)}
              </motion.div>
            ))}
          </div>

          {/* Legend */}
          <div className="flex flex-wrap gap-4 text-sm">
            <div className="flex items-center space-x-2">
              <div className="w-4 h-4 bg-blue-500 rounded"></div>
              <span>Agent 1</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-4 h-4 bg-red-500 rounded"></div>
              <span>Agent 2</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-4 h-4 bg-yellow-500 rounded"></div>
              <span>Agent 3</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-4 h-4 bg-purple-500 rounded"></div>
              <span>Agent 4</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-4 h-4 bg-green-400 rounded"></div>
              <span>Goal</span>
            </div>
          </div>

          {/* Training Status */}
          <div className="bg-blue-50 p-3 rounded-lg">
            <div className="flex justify-between items-center text-sm">
              <span className="text-blue-700">Step: {currentStep + 1}/100</span>
              <span className="text-blue-700">
                Status: {isTraining ? 'Training...' : 'Ready'}
              </span>
            </div>
            {trainingData && trainingData.episodes && (
              <div className="mt-2 text-sm text-blue-600">
                Episodes completed: {trainingData.episodes.length}
              </div>
            )}
          </div>
        </div>
      ) : (
        <div className="text-center py-8 text-gray-500">
          <EyeIcon className="h-12 w-12 mx-auto mb-4 text-gray-300" />
          <p>Start training to see environment visualization</p>
        </div>
      )}
    </motion.div>
  );
};

export default EnvironmentVisualizer;
