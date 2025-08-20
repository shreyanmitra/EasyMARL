/**
 * API Service for EasyMARL React Frontend
 * 
 * Handles communication with the Python backend for training,
 * data retrieval, and configuration management. This service
 * bridges the React frontend with the existing Python controllers.
 */

// Configure API base URL for local and Codespaces environments only
const getApiBaseUrl = () => {
  // Check if running in GitHub Codespaces
  const hostname = window.location.hostname;
  
  if (hostname.includes('app.github.dev')) {
    // Extract codespace name from frontend URL
    const codespaceName = hostname.split('-')[0];
    const backendUrl = `https://${codespaceName}-5000.app.github.dev/api`;
    console.log('🎓 Running in Codespaces, using backend:', backendUrl);
    return backendUrl;
  }
  
  // Custom API URL via environment variable (for local development)
  if (process.env.REACT_APP_API_URL) {
    return process.env.REACT_APP_API_URL;
  }
  
  // Local development fallback
  return 'http://localhost:5000/api';
};

const API_BASE_URL = getApiBaseUrl();

// Check if running in demo/read-only mode
const DEMO_MODE = process.env.REACT_APP_DEMO_MODE === 'true';
const READ_ONLY = process.env.REACT_APP_READ_ONLY === 'true';

class TrainingAPI {
  /**
   * Start training with specified configuration
   * 
   * @param {Object} config - Training configuration
   * @param {string} config.environment - Environment name
   * @param {string} config.algorithm - Algorithm name
   * @param {string} config.controllerType - 'simple' or 'modern'
   * @param {number} config.maxEpisodes - Maximum episodes to train
   * @param {number} config.learningRate - Learning rate
   * @param {boolean} config.useWandB - Enable WandB logging
   * @returns {Promise<Object>} Response with success status
   */
  async startTraining(config) {
    try {
      const response = await fetch(`${API_BASE_URL}/training/start`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(config),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error starting training:', error);
      throw error;
    }
  }

  /**
   * Stop current training session
   * 
   * @returns {Promise<Object>} Response with success status
   */
  async stopTraining() {
    try {
      const response = await fetch(`${API_BASE_URL}/training/stop`, {
        method: 'POST',
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error stopping training:', error);
      throw error;
    }
  }

  /**
   * Get current training progress
   * 
   * @returns {Promise<Object>} Training progress data
   */
  async getTrainingProgress() {
    try {
      const response = await fetch(`${API_BASE_URL}/training/progress`);

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error fetching training progress:', error);
      throw error;
    }
  }

  /**
   * Get available algorithms with descriptions
   * 
   * @returns {Promise<Array>} List of algorithms
   */
  async getAlgorithms() {
    try {
      const response = await fetch(`${API_BASE_URL}/algorithms`);

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error fetching algorithms:', error);
      throw error;
    }
  }

  /**
   * Get available environments with descriptions
   * 
   * @returns {Promise<Array>} List of environments
   */
  async getEnvironments() {
    try {
      const response = await fetch(`${API_BASE_URL}/environments`);

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error fetching environments:', error);
      throw error;
    }
  }

  /**
   * Download training data as JSON
   * 
   * @param {string} sessionId - Training session ID
   * @returns {Promise<Blob>} Training data blob
   */
  async downloadTrainingData(sessionId) {
    try {
      const response = await fetch(`${API_BASE_URL}/training/download/${sessionId}`);

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.blob();
    } catch (error) {
      console.error('Error downloading training data:', error);
      throw error;
    }
  }

  /**
   * Get training video information
   * 
   * @param {string} sessionId - Training session ID
   * @returns {Promise<Object>} Video information with download URL
   */
  async getTrainingVideo(sessionId) {
    try {
      const response = await fetch(`${API_BASE_URL}/training/video/${sessionId}`);

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error getting training video:', error);
      throw error;
    }
  }

  /**
   * Download training video
   * 
   * @param {string} sessionId - Training session ID
   * @returns {Promise<Blob>} Video blob
   */
  async downloadTrainingVideo(sessionId) {
    try {
      const response = await fetch(`${API_BASE_URL}/training/download-video/${sessionId}`);

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.blob();
    } catch (error) {
      console.error('Error downloading training video:', error);
      throw error;
    }
  }
}

export const trainingAPI = new TrainingAPI();
