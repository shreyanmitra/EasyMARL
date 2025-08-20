import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { PlayIcon, StopIcon, DownloadIcon, VideoCameraIcon } from '@heroicons/react/outline';
import toast from 'react-hot-toast';
import TrainingChart from '../components/TrainingChart';
import EnvironmentVisualizer from '../components/EnvironmentVisualizer';
import { trainingAPI } from '../services/api';

/**
 * Training Page Component
 * 
 * This page provides the same functionality as the Python GUI training tab:
 * - Algorithm and environment selection
 * - Controller type selection (Simple vs Modern)
 * - Real-time training monitoring
 * - Training parameter configuration
 * - Live progress visualization
 */
const TrainingPage = () => {
  // Training state management
  const [isTraining, setIsTraining] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const [trainingData, setTrainingData] = useState({
    episodes: [],
    rewards: [],
    lengths: [],
    currentEpisode: 0
  });

  // Configuration state
  const [config, setConfig] = useState({
    environment: 'MultiGrid-Cluttered-Fixed-15x15',
    algorithm: 'qmix',
    controllerType: 'simple', // Default to simple for beginners
    maxEpisodes: 1000,
    learningRate: 0.001,
    useWandB: false
  });

  // Available options (same as Python GUI)
  const algorithms = [
    { value: 'qmix', label: 'QMIX', category: 'Value Decomposition', difficulty: 'Beginner' },
    { value: 'vdn', label: 'VDN', category: 'Value Decomposition', difficulty: 'Beginner' },
    { value: 'qtran', label: 'QTRAN', category: 'Value Decomposition', difficulty: 'Advanced' },
    { value: 'mappo', label: 'MAPPO', category: 'Actor-Critic', difficulty: 'Intermediate' },
    { value: 'maddpg', label: 'MADDPG', category: 'Actor-Critic', difficulty: 'Intermediate' },
    { value: 'coma', label: 'COMA', category: 'Actor-Critic', difficulty: 'Advanced' },
    // ... include all 21+ algorithms from the Python version
  ];

  const environments = [
    { value: 'MultiGrid-Cluttered-Fixed-15x15', label: 'MultiGrid Cluttered (15x15)' },
    { value: 'MultiGrid-Empty-8x8', label: 'MultiGrid Empty (8x8)' },
    { value: 'MultiGrid-FourRooms', label: 'MultiGrid Four Rooms' }
  ];

  const controllerTypes = [
    { 
      value: 'simple', 
      label: 'Simple (Beginner-friendly)', 
      description: 'Structured like original metacontroller, easier to understand'
    },
    { 
      value: 'modern', 
      label: 'Modern (Advanced features)', 
      description: 'Full-featured with advanced metrics and comprehensive logging'
    }
  ];

  /**
   * Start training with selected configuration
   * Communicates with Python backend API to begin training
   */
  const startTraining = async () => {
    try {
      setIsTraining(true);
      toast.success(`Starting training with ${config.algorithm.toUpperCase()} on ${config.environment}`);
      
      // Send training request to Python backend
      const response = await trainingAPI.startTraining(config);
      
      if (response.success) {
        setSessionId(response.sessionId || Date.now().toString()); // Fallback session ID
        // Start polling for training updates
        pollTrainingProgress();
      } else {
        throw new Error(response.error || 'Failed to start training');
      }
    } catch (error) {
      toast.error(`Training failed: ${error.message}`);
      setIsTraining(false);
    }
  };

  /**
   * Stop current training session
   */
  const stopTraining = async () => {
    try {
      await trainingAPI.stopTraining();
      setIsTraining(false);
      toast.success('Training stopped');
    } catch (error) {
      toast.error(`Failed to stop training: ${error.message}`);
    }
  };

  /**
   * Poll backend for training progress updates
   * This replaces the real-time updates from the Python GUI
   */
  const pollTrainingProgress = () => {
    const interval = setInterval(async () => {
      try {
        const progress = await trainingAPI.getTrainingProgress();
        
        if (progress.isTraining) {
          setTrainingData(prev => ({
            episodes: [...prev.episodes, progress.episode],
            rewards: [...prev.rewards, progress.reward],
            lengths: [...prev.lengths, progress.length],
            currentEpisode: progress.episode
          }));
        } else {
          clearInterval(interval);
          setIsTraining(false);
          toast.success('Training completed!');
        }
      } catch (error) {
        console.error('Failed to fetch training progress:', error);
        clearInterval(interval);
        setIsTraining(false);
      }
    }, 2000); // Update every 2 seconds like Python GUI

    return interval;
  };

  /**
   * Download training data as JSON
   * Same functionality as Python GUI download feature
   */
  const downloadTrainingData = () => {
    const dataToDownload = {
      config,
      trainingData,
      timestamp: new Date().toISOString(),
      framework: 'EasyMARL React'
    };

    const blob = new Blob([JSON.stringify(dataToDownload, null, 2)], {
      type: 'application/json'
    });
    
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `easymarl_training_${config.algorithm}_${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
    toast.success('Training data downloaded!');
  };

  /**
   * Download training video
   * Same functionality as Python GUI video download feature
   */
  const downloadTrainingVideo = async () => {
    if (!sessionId) {
      toast.error('No training session found');
      return;
    }

    try {
      toast.loading('Generating training video...');
      
      // Get video information first
      const videoInfo = await trainingAPI.getTrainingVideo(sessionId);
      
      if (videoInfo.success) {
        // Download the video file
        const videoBlob = await trainingAPI.downloadTrainingVideo(sessionId);
        
        const url = URL.createObjectURL(videoBlob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `easymarl_training_video_${config.algorithm}_${Date.now()}.mp4`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        toast.dismiss();
        toast.success('Training video downloaded!');
      } else {
        throw new Error(videoInfo.error || 'Failed to generate video');
      }
    } catch (error) {
      toast.dismiss();
      toast.error(`Video download failed: ${error.message}`);
    }
  };

  return (
    <div className="max-w-7xl mx-auto space-y-8">
      {/* Page Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center"
      >
        <h1 className="text-4xl font-bold text-gray-900 mb-4">
          🎯 MARL Training Center
        </h1>
        <p className="text-xl text-gray-600">
          Train state-of-the-art multi-agent reinforcement learning algorithms
        </p>
      </motion.div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Configuration Panel */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          className="lg:col-span-1 space-y-6"
        >
          <div className="bg-white rounded-xl shadow-lg p-6">
            <h2 className="text-2xl font-semibold mb-6 text-gray-800">
              ⚙️ Configuration
            </h2>

            {/* Environment Selection */}
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Environment
                </label>
                <select
                  value={config.environment}
                  onChange={(e) => setConfig(prev => ({ ...prev, environment: e.target.value }))}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  disabled={isTraining}
                >
                  {environments.map(env => (
                    <option key={env.value} value={env.value}>
                      {env.label}
                    </option>
                  ))}
                </select>
              </div>

              {/* Algorithm Selection */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Algorithm
                </label>
                <select
                  value={config.algorithm}
                  onChange={(e) => setConfig(prev => ({ ...prev, algorithm: e.target.value }))}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  disabled={isTraining}
                >
                  {algorithms.map(alg => (
                    <option key={alg.value} value={alg.value}>
                      {alg.label} ({alg.difficulty})
                    </option>
                  ))}
                </select>
              </div>

              {/* Controller Type Selection - NEW FEATURE */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Controller Type
                </label>
                <div className="space-y-2">
                  {controllerTypes.map(controller => (
                    <label key={controller.value} className="flex items-start space-x-3">
                      <input
                        type="radio"
                        name="controllerType"
                        value={controller.value}
                        checked={config.controllerType === controller.value}
                        onChange={(e) => setConfig(prev => ({ ...prev, controllerType: e.target.value }))}
                        disabled={isTraining}
                        className="mt-1"
                      />
                      <div>
                        <div className="font-medium text-gray-900">{controller.label}</div>
                        <div className="text-sm text-gray-600">{controller.description}</div>
                      </div>
                    </label>
                  ))}
                </div>
              </div>

              {/* Training Parameters */}
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Max Episodes
                  </label>
                  <input
                    type="number"
                    value={config.maxEpisodes}
                    onChange={(e) => setConfig(prev => ({ ...prev, maxEpisodes: parseInt(e.target.value) }))}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    min="10"
                    max="10000"
                    disabled={isTraining}
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Learning Rate
                  </label>
                  <input
                    type="number"
                    value={config.learningRate}
                    onChange={(e) => setConfig(prev => ({ ...prev, learningRate: parseFloat(e.target.value) }))}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    min="0.0001"
                    max="1.0"
                    step="0.0001"
                    disabled={isTraining}
                  />
                </div>
              </div>

              {/* WandB Integration */}
              <label className="flex items-center space-x-3">
                <input
                  type="checkbox"
                  checked={config.useWandB}
                  onChange={(e) => setConfig(prev => ({ ...prev, useWandB: e.target.checked }))}
                  disabled={isTraining}
                  className="rounded"
                />
                <span className="text-sm font-medium text-gray-700">
                  Enable WandB Logging
                </span>
              </label>
            </div>

            {/* Training Controls */}
            <div className="mt-6 space-y-3">
              {!isTraining ? (
                <button
                  onClick={startTraining}
                  className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 px-4 rounded-lg flex items-center justify-center space-x-2 transition-colors"
                >
                  <PlayIcon className="h-5 w-5" />
                  <span>🚀 Start Training</span>
                </button>
              ) : (
                <button
                  onClick={stopTraining}
                  className="w-full bg-red-600 hover:bg-red-700 text-white font-semibold py-3 px-4 rounded-lg flex items-center justify-center space-x-2 transition-colors"
                >
                  <StopIcon className="h-5 w-5" />
                  <span>⏹️ Stop Training</span>
                </button>
              )}

              {trainingData.episodes.length > 0 && (
                <>
                  <button
                    onClick={downloadTrainingData}
                    className="w-full bg-green-600 hover:bg-green-700 text-white font-semibold py-3 px-4 rounded-lg flex items-center justify-center space-x-2 transition-colors"
                  >
                    <DownloadIcon className="h-5 w-5" />
                    <span>📊 Download Data</span>
                  </button>
                  
                  {sessionId && (
                    <button
                      onClick={downloadTrainingVideo}
                      className="w-full bg-purple-600 hover:bg-purple-700 text-white font-semibold py-3 px-4 rounded-lg flex items-center justify-center space-x-2 transition-colors"
                    >
                      <VideoCameraIcon className="h-5 w-5" />
                      <span>🎥 Download Video</span>
                    </button>
                  )}
                </>
              )}
            </div>

            {/* Training Status */}
            <div className="mt-4 p-3 bg-gray-100 rounded-lg">
              <div className="text-sm font-medium text-gray-700">
                Status: {isTraining ? 'Training in progress...' : 'Ready to train'}
              </div>
              {isTraining && (
                <div className="text-sm text-gray-600">
                  Episode: {trainingData.currentEpisode} / {config.maxEpisodes}
                </div>
              )}
            </div>
          </div>
        </motion.div>

        {/* Visualization Panel */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          className="lg:col-span-2 space-y-6"
        >
          {/* Training Progress Charts */}
          <div className="bg-white rounded-xl shadow-lg p-6">
            <h2 className="text-2xl font-semibold mb-6 text-gray-800">
              📈 Training Progress
            </h2>
            <TrainingChart data={trainingData} />
          </div>

          {/* Environment Visualization */}
          <div className="bg-white rounded-xl shadow-lg p-6">
            <h2 className="text-2xl font-semibold mb-6 text-gray-800">
              🌍 Environment Visualization
            </h2>
            <EnvironmentVisualizer 
              environment={config.environment}
              isTraining={isTraining}
            />
          </div>
        </motion.div>
      </div>
    </div>
  );
};

export default TrainingPage;
