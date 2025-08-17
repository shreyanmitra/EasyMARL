import React, { useState, useEffect } from 'react';
import '../styles/ParameterConfigurator.css';

const ParameterConfigurator = ({ algorithm, onConfigChange, initialConfig = null }) => {
  const [parameters, setParameters] = useState({});
  const [template, setTemplate] = useState(null);
  const [config, setConfig] = useState({});
  const [activeTab, setActiveTab] = useState('learning');
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [previewMode, setPreviewMode] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (algorithm) {
      fetchAlgorithmParameters();
      fetchAlgorithmTemplate();
    }
  }, [algorithm]);

  useEffect(() => {
    if (initialConfig) {
      setConfig(initialConfig);
    }
  }, [initialConfig]);

  const fetchAlgorithmParameters = async () => {
    try {
      setLoading(true);
      const response = await fetch(`/api/config/algorithm-parameters/${algorithm}`);
      const data = await response.json();
      
      if (data.success) {
        setParameters(data.parameters);
        initializeConfig(data.parameters);
      } else {
        setError(data.error);
      }
    } catch (err) {
      setError('Failed to fetch algorithm parameters');
    } finally {
      setLoading(false);
    }
  };

  const fetchAlgorithmTemplate = async () => {
    try {
      const response = await fetch(`/api/config/algorithm-template/${algorithm}`);
      const data = await response.json();
      
      if (data.success) {
        setTemplate(data.template);
      }
    } catch (err) {
      console.error('Failed to fetch algorithm template:', err);
    }
  };

  const initializeConfig = (params) => {
    const initialConfig = {};
    
    Object.keys(params).forEach(category => {
      if (typeof params[category] === 'object' && params[category] !== null) {
        initialConfig[category] = {};
        Object.keys(params[category]).forEach(param => {
          const paramInfo = params[category][param];
          initialConfig[category][param] = paramInfo.default;
        });
      }
    });
    
    setConfig(initialConfig);
    if (onConfigChange) {
      onConfigChange(initialConfig);
    }
  };

  const handleParameterChange = (category, parameter, value) => {
    const updatedConfig = {
      ...config,
      [category]: {
        ...config[category],
        [parameter]: value
      }
    };
    
    setConfig(updatedConfig);
    if (onConfigChange) {
      onConfigChange(updatedConfig);
    }
  };

  const renderParameterInput = (category, parameter, paramInfo) => {
    const value = config[category]?.[parameter] || paramInfo.default;
    
    switch (paramInfo.type) {
      case 'float':
        return (
          <div className="parameter-input">
            <label>{parameter.replace(/_/g, ' ').toUpperCase()}</label>
            <div className="input-group">
              <input
                type="number"
                step={paramInfo.log_scale ? "any" : "0.001"}
                min={paramInfo.range?.[0]}
                max={paramInfo.range?.[1]}
                value={value}
                onChange={(e) => handleParameterChange(category, parameter, parseFloat(e.target.value))}
              />
              {paramInfo.range && (
                <input
                  type="range"
                  min={paramInfo.range[0]}
                  max={paramInfo.range[1]}
                  step={paramInfo.log_scale ? "any" : "0.001"}
                  value={value}
                  onChange={(e) => handleParameterChange(category, parameter, parseFloat(e.target.value))}
                  className="range-slider"
                />
              )}
            </div>
            {paramInfo.range && (
              <span className="range-info">
                Range: {paramInfo.range[0]} - {paramInfo.range[1]}
              </span>
            )}
          </div>
        );
      
      case 'int':
        return (
          <div className="parameter-input">
            <label>{parameter.replace(/_/g, ' ').toUpperCase()}</label>
            <div className="input-group">
              <input
                type="number"
                step="1"
                min={paramInfo.range?.[0]}
                max={paramInfo.range?.[1]}
                value={value}
                onChange={(e) => handleParameterChange(category, parameter, parseInt(e.target.value))}
              />
              {paramInfo.range && (
                <input
                  type="range"
                  min={paramInfo.range[0]}
                  max={paramInfo.range[1]}
                  step="1"
                  value={value}
                  onChange={(e) => handleParameterChange(category, parameter, parseInt(e.target.value))}
                  className="range-slider"
                />
              )}
            </div>
            {paramInfo.range && (
              <span className="range-info">
                Range: {paramInfo.range[0]} - {paramInfo.range[1]}
              </span>
            )}
          </div>
        );
      
      case 'boolean':
        return (
          <div className="parameter-input">
            <label>
              <input
                type="checkbox"
                checked={value}
                onChange={(e) => handleParameterChange(category, parameter, e.target.checked)}
              />
              {parameter.replace(/_/g, ' ').toUpperCase()}
            </label>
          </div>
        );
      
      case 'categorical':
        return (
          <div className="parameter-input">
            <label>{parameter.replace(/_/g, ' ').toUpperCase()}</label>
            <select
              value={value}
              onChange={(e) => handleParameterChange(category, parameter, e.target.value)}
            >
              {paramInfo.options.map(option => (
                <option key={option} value={option}>{option}</option>
              ))}
            </select>
          </div>
        );
      
      case 'list_int':
        return (
          <div className="parameter-input">
            <label>{parameter.replace(/_/g, ' ').toUpperCase()}</label>
            <select
              value={JSON.stringify(value)}
              onChange={(e) => handleParameterChange(category, parameter, JSON.parse(e.target.value))}
            >
              {paramInfo.options.map(option => (
                <option key={JSON.stringify(option)} value={JSON.stringify(option)}>
                  [{option.join(', ')}]
                </option>
              ))}
            </select>
          </div>
        );
      
      default:
        return (
          <div className="parameter-input">
            <label>{parameter.replace(/_/g, ' ').toUpperCase()}</label>
            <input
              type="text"
              value={value}
              onChange={(e) => handleParameterChange(category, parameter, e.target.value)}
            />
          </div>
        );
    }
  };

  const renderParameterCategory = (categoryName, categoryParams) => {
    if (!categoryParams || typeof categoryParams !== 'object') return null;
    
    return (
      <div className="parameter-category">
        <h3>{categoryName.replace(/_/g, ' ').toUpperCase()}</h3>
        <div className="parameters-grid">
          {Object.entries(categoryParams).map(([parameter, paramInfo]) => (
            <div key={parameter} className="parameter-item">
              {renderParameterInput(categoryName, parameter, paramInfo)}
            </div>
          ))}
        </div>
      </div>
    );
  };

  const renderAlgorithmInfo = () => {
    if (!template) return null;
    
    return (
      <div className="algorithm-info">
        <h3>Algorithm Information</h3>
        <div className="info-grid">
          <div className="info-item">
            <strong>Type:</strong> {template.algorithm_type}
          </div>
          <div className="info-item">
            <strong>Paradigm:</strong> {template.learning_paradigm?.replace(/_/g, ' ')}
          </div>
          <div className="info-item">
            <strong>Action Space:</strong> {template.action_space}
          </div>
          {template.tabular && (
            <div className="info-item">
              <strong>Method:</strong> Tabular
            </div>
          )}
        </div>
      </div>
    );
  };

  const renderMemorySettings = () => {
    return (
      <div className="memory-settings">
        <h3>Memory & Experience Settings</h3>
        <div className="memory-grid">
          <div className="memory-section">
            <h4>Memory Configuration</h4>
            <div className="checkbox-group">
              <label>
                <input
                  type="checkbox"
                  checked={config.memory_parameters?.shared_memory || false}
                  onChange={(e) => handleParameterChange('memory_parameters', 'shared_memory', e.target.checked)}
                />
                Shared Memory Between Agents
              </label>
              <label>
                <input
                  type="checkbox"
                  checked={config.memory_parameters?.experience_sharing || false}
                  onChange={(e) => handleParameterChange('memory_parameters', 'experience_sharing', e.target.checked)}
                />
                Experience Sharing
              </label>
              <label>
                <input
                  type="checkbox"
                  checked={config.memory_parameters?.use_prioritized || false}
                  onChange={(e) => handleParameterChange('memory_parameters', 'use_prioritized', e.target.checked)}
                />
                Prioritized Experience Replay
              </label>
            </div>
          </div>
        </div>
      </div>
    );
  };

  const renderCommunicationSettings = () => {
    return (
      <div className="communication-settings">
        <h3>Communication Settings</h3>
        <div className="communication-grid">
          <div className="communication-section">
            <h4>Communication Protocol</h4>
            <div className="checkbox-group">
              <label>
                <input
                  type="checkbox"
                  checked={config.communication_parameters?.enabled || false}
                  onChange={(e) => handleParameterChange('communication_parameters', 'enabled', e.target.checked)}
                />
                Enable Communication
              </label>
              <label>
                <input
                  type="checkbox"
                  checked={config.communication_parameters?.learnable_communication || false}
                  onChange={(e) => handleParameterChange('communication_parameters', 'learnable_communication', e.target.checked)}
                />
                Learnable Communication
              </label>
              <label>
                <input
                  type="checkbox"
                  checked={config.communication_parameters?.differentiable || false}
                  onChange={(e) => handleParameterChange('communication_parameters', 'differentiable', e.target.checked)}
                />
                Differentiable Communication
              </label>
            </div>
          </div>
        </div>
      </div>
    );
  };

  const renderCoordinationSettings = () => {
    return (
      <div className="coordination-settings">
        <h3>Multi-Agent Coordination</h3>
        <div className="coordination-grid">
          <div className="coordination-section">
            <h4>Training Paradigm</h4>
            <div className="checkbox-group">
              <label>
                <input
                  type="checkbox"
                  checked={config.coordination_parameters?.centralized_training || false}
                  onChange={(e) => handleParameterChange('coordination_parameters', 'centralized_training', e.target.checked)}
                />
                Centralized Training
              </label>
              <label>
                <input
                  type="checkbox"
                  checked={config.coordination_parameters?.decentralized_execution || false}
                  onChange={(e) => handleParameterChange('coordination_parameters', 'decentralized_execution', e.target.checked)}
                />
                Decentralized Execution
              </label>
              <label>
                <input
                  type="checkbox"
                  checked={config.coordination_parameters?.observation_sharing || false}
                  onChange={(e) => handleParameterChange('coordination_parameters', 'observation_sharing', e.target.checked)}
                />
                Observation Sharing
              </label>
              <label>
                <input
                  type="checkbox"
                  checked={config.coordination_parameters?.action_sharing || false}
                  onChange={(e) => handleParameterChange('coordination_parameters', 'action_sharing', e.target.checked)}
                />
                Action Sharing
              </label>
              <label>
                <input
                  type="checkbox"
                  checked={config.coordination_parameters?.reward_sharing || false}
                  onChange={(e) => handleParameterChange('coordination_parameters', 'reward_sharing', e.target.checked)}
                />
                Reward Sharing
              </label>
            </div>
          </div>
        </div>
      </div>
    );
  };

  const renderConfigPreview = () => {
    return (
      <div className="config-preview">
        <h3>Configuration Preview</h3>
        <pre>{JSON.stringify(config, null, 2)}</pre>
      </div>
    );
  };

  if (loading) {
    return <div className="loading">Loading algorithm parameters...</div>;
  }

  if (error) {
    return <div className="error">Error: {error}</div>;
  }

  return (
    <div className="parameter-configurator">
      <div className="configurator-header">
        <h2>Configure {algorithm} Parameters</h2>
        <div className="header-controls">
          <button
            className={`toggle-btn ${showAdvanced ? 'active' : ''}`}
            onClick={() => setShowAdvanced(!showAdvanced)}
          >
            {showAdvanced ? 'Hide' : 'Show'} Advanced Settings
          </button>
          <button
            className={`toggle-btn ${previewMode ? 'active' : ''}`}
            onClick={() => setPreviewMode(!previewMode)}
          >
            {previewMode ? 'Hide' : 'Show'} Preview
          </button>
        </div>
      </div>

      {renderAlgorithmInfo()}

      <div className="configurator-content">
        <div className="tab-navigation">
          <button
            className={`tab ${activeTab === 'learning' ? 'active' : ''}`}
            onClick={() => setActiveTab('learning')}
          >
            Learning
          </button>
          <button
            className={`tab ${activeTab === 'network' ? 'active' : ''}`}
            onClick={() => setActiveTab('network')}
          >
            Network
          </button>
          <button
            className={`tab ${activeTab === 'memory' ? 'active' : ''}`}
            onClick={() => setActiveTab('memory')}
          >
            Memory
          </button>
          <button
            className={`tab ${activeTab === 'coordination' ? 'active' : ''}`}
            onClick={() => setActiveTab('coordination')}
          >
            Coordination
          </button>
          <button
            className={`tab ${activeTab === 'communication' ? 'active' : ''}`}
            onClick={() => setActiveTab('communication')}
          >
            Communication
          </button>
          {showAdvanced && (
            <button
              className={`tab ${activeTab === 'advanced' ? 'active' : ''}`}
              onClick={() => setActiveTab('advanced')}
            >
              Advanced
            </button>
          )}
        </div>

        <div className="tab-content">
          {activeTab === 'learning' && (
            <div>
              {renderParameterCategory('learning_parameters', parameters.learning_parameters)}
              {renderParameterCategory('exploration_parameters', parameters.exploration_parameters)}
              {parameters.algorithm_specific && renderParameterCategory('algorithm_specific', parameters.algorithm_specific)}
            </div>
          )}
          
          {activeTab === 'network' && (
            renderParameterCategory('network_parameters', parameters.network_parameters)
          )}
          
          {activeTab === 'memory' && (
            <div>
              {renderParameterCategory('memory_parameters', parameters.memory_parameters)}
              {renderMemorySettings()}
            </div>
          )}
          
          {activeTab === 'coordination' && (
            <div>
              {renderParameterCategory('coordination_parameters', parameters.coordination_parameters)}
              {renderCoordinationSettings()}
            </div>
          )}
          
          {activeTab === 'communication' && (
            <div>
              {renderParameterCategory('communication_parameters', parameters.communication_parameters)}
              {renderCommunicationSettings()}
            </div>
          )}
          
          {activeTab === 'advanced' && showAdvanced && (
            renderParameterCategory('advanced_parameters', parameters.advanced_parameters)
          )}
        </div>

        {previewMode && renderConfigPreview()}
      </div>
    </div>
  );
};

export default ParameterConfigurator;
