/**
 * API Service for EasyMARL React Frontend
 * 
 * Handles communication with the Python backend for training,
 * data retrieval, and configuration management. This service
 * bridges the React frontend with the existing Python controllers.
 */

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

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
}

export const trainingAPI = new TrainingAPI();
