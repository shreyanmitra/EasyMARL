import React, { useState, useEffect } from 'react';
import { 
  Grid, 
  Save, 
  Download, 
  Upload, 
  Eye, 
  Settings,
  Plus,
  Trash2,
  Play,
  Square
} from 'lucide-react';

const EnvironmentBuilder = () => {
  const [templates, setTemplates] = useState({});
  const [selectedTemplate, setSelectedTemplate] = useState(null);
  const [environmentConfig, setEnvironmentConfig] = useState({
    name: 'Custom-Environment',
    width: 8,
    height: 8,
    n_agents: 2,
    max_steps: 100,
    agent_view_size: 7,
    see_through_walls: true,
    description: '',
    agents: [
      { x: 1, y: 1, direction: 0 },
      { x: 6, y: 6, direction: 2 }
    ],
    objects: []
  });
  const [selectedTool, setSelectedTool] = useState('wall');
  const [selectedColor, setSelectedColor] = useState('red');
  const [grid, setGrid] = useState([]);
  const [preview, setPreview] = useState('');
  const [saving, setSaving] = useState(false);

  // Initialize grid
  useEffect(() => {
    initializeGrid();
  }, [environmentConfig.width, environmentConfig.height]);

  // Load templates on component mount
  useEffect(() => {
    loadTemplates();
  }, []);

  const initializeGrid = () => {
    const newGrid = Array(environmentConfig.height).fill(null).map(() => 
      Array(environmentConfig.width).fill(null)
    );
    
    // Add border walls
    for (let y = 0; y < environmentConfig.height; y++) {
      for (let x = 0; x < environmentConfig.width; x++) {
        if (x === 0 || x === environmentConfig.width - 1 || 
            y === 0 || y === environmentConfig.height - 1) {
          newGrid[y][x] = { type: 'wall', color: 'grey' };
        }
      }
    }
    
    // Add objects from config
    environmentConfig.objects.forEach(obj => {
      if (obj.x < environmentConfig.width && obj.y < environmentConfig.height) {
        newGrid[obj.y][obj.x] = obj;
      }
    });
    
    setGrid(newGrid);
  };

  const loadTemplates = async () => {
    try {
      const response = await fetch('/api/environment-builder/templates');
      const data = await response.json();
      if (data.success) {
        setTemplates(data.templates);
      }
    } catch (error) {
      console.error('Failed to load templates:', error);
    }
  };

  const handleCellClick = (x, y) => {
    if (selectedTool === 'eraser') {
      removeObject(x, y);
    } else {
      placeObject(x, y);
    }
  };

  const placeObject = (x, y) => {
    // Don't allow placing on border walls
    if (x === 0 || x === environmentConfig.width - 1 || 
        y === 0 || y === environmentConfig.height - 1) {
      return;
    }

    const newObject = {
      type: selectedTool,
      x: x,
      y: y,
      color: selectedColor,
      locked: selectedTool === 'door' ? true : false
    };

    const newObjects = environmentConfig.objects.filter(obj => 
      !(obj.x === x && obj.y === y)
    );
    newObjects.push(newObject);

    setEnvironmentConfig(prev => ({
      ...prev,
      objects: newObjects
    }));
  };

  const removeObject = (x, y) => {
    // Don't allow removing border walls
    if (x === 0 || x === environmentConfig.width - 1 || 
        y === 0 || y === environmentConfig.height - 1) {
      return;
    }

    const newObjects = environmentConfig.objects.filter(obj => 
      !(obj.x === x && obj.y === y)
    );

    setEnvironmentConfig(prev => ({
      ...prev,
      objects: newObjects
    }));
  };

  const saveEnvironment = async () => {
    setSaving(true);
    try {
      const response = await fetch('/api/environment-builder/create', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(environmentConfig)
      });
      
      const data = await response.json();
      if (data.success) {
        alert(`Environment "${environmentConfig.name}" saved successfully!`);
        generatePreview(data.environment_id);
      } else {
        alert(`Error saving environment: ${data.error}`);
      }
    } catch (error) {
      alert(`Error saving environment: ${error.message}`);
    }
    setSaving(false);
  };

  const generatePreview = async (envId) => {
    try {
      const response = await fetch(`/api/environment-builder/preview/${envId}`);
      const data = await response.json();
      if (data.success) {
        setPreview(data.preview);
      }
    } catch (error) {
      console.error('Failed to generate preview:', error);
    }
  };

  const exportEnvironment = async (format) => {
    try {
      // First save the environment
      const saveResponse = await fetch('/api/environment-builder/create', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(environmentConfig)
      });
      
      const saveData = await saveResponse.json();
      if (!saveData.success) {
        alert(`Error saving environment: ${saveData.error}`);
        return;
      }

      // Then export it
      const exportResponse = await fetch(
        `/api/environment-builder/export/${saveData.environment_id}?format=${format}`
      );
      const exportData = await exportResponse.json();
      
      if (exportData.success) {
        // Download the file
        const blob = new Blob([exportData.content], { 
          type: exportData.content_type 
        });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = exportData.filename;
        a.click();
        URL.revokeObjectURL(url);
      } else {
        alert(`Error exporting environment: ${exportData.error}`);
      }
    } catch (error) {
      alert(`Error exporting environment: ${error.message}`);
    }
  };

  const loadTemplate = (template) => {
    setEnvironmentConfig(prev => ({
      ...prev,
      name: template.name,
      width: template.grid_size[0],
      height: template.grid_size[1],
      n_agents: template.n_agents,
      objects: template.objects || [],
      description: template.description
    }));
    setSelectedTemplate(template);
  };

  const getCellDisplay = (cell, x, y) => {
    // Check for agents first
    const agent = environmentConfig.agents.find(a => a.x === x && a.y === y);
    if (agent) {
      const agentIndex = environmentConfig.agents.indexOf(agent);
      return { symbol: `A${agentIndex + 1}`, className: 'bg-blue-500 text-white' };
    }

    // Check for objects
    if (cell) {
      const symbols = {
        wall: '█',
        door: '🚪',
        key: '🔑',
        goal: '🎯',
        ball: '⚽',
        box: '📦',
        lava: '🔥'
      };
      
      const colors = {
        red: 'bg-red-500',
        green: 'bg-green-500',
        blue: 'bg-blue-500',
        yellow: 'bg-yellow-500',
        purple: 'bg-purple-500',
        grey: 'bg-gray-500'
      };

      return {
        symbol: symbols[cell.type] || '?',
        className: `${colors[cell.color] || 'bg-gray-500'} text-white`
      };
    }

    return { symbol: '', className: 'bg-white hover:bg-gray-100' };
  };

  const tools = [
    { id: 'wall', name: 'Wall', icon: '█' },
    { id: 'door', name: 'Door', icon: '🚪' },
    { id: 'key', name: 'Key', icon: '🔑' },
    { id: 'goal', name: 'Goal', icon: '🎯' },
    { id: 'ball', name: 'Ball', icon: '⚽' },
    { id: 'box', name: 'Box', icon: '📦' },
    { id: 'lava', name: 'Lava', icon: '🔥' },
    { id: 'eraser', name: 'Eraser', icon: '🗑️' }
  ];

  const colors = ['red', 'green', 'blue', 'yellow', 'purple', 'grey'];

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-6">
      <div className="max-w-7xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            🏗️ Environment Builder
          </h1>
          <p className="text-xl text-gray-600">
            Create custom MultiGrid environments with an intuitive visual interface
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          
          {/* Left Panel - Tools and Config */}
          <div className="lg:col-span-1 space-y-6">
            
            {/* Tools */}
            <div className="bg-white rounded-lg shadow-lg p-6">
              <h3 className="text-lg font-semibold mb-4 flex items-center">
                <Settings className="w-5 h-5 mr-2" />
                Tools
              </h3>
              <div className="grid grid-cols-2 gap-2">
                {tools.map(tool => (
                  <button
                    key={tool.id}
                    onClick={() => setSelectedTool(tool.id)}
                    className={`p-3 rounded-lg border-2 transition-all ${
                      selectedTool === tool.id
                        ? 'border-blue-500 bg-blue-50 text-blue-700'
                        : 'border-gray-200 hover:border-gray-300'
                    }`}
                  >
                    <div className="text-lg">{tool.icon}</div>
                    <div className="text-xs">{tool.name}</div>
                  </button>
                ))}
              </div>
              
              <div className="mt-4">
                <label className="block text-sm font-medium mb-2">Color</label>
                <div className="grid grid-cols-3 gap-2">
                  {colors.map(color => (
                    <button
                      key={color}
                      onClick={() => setSelectedColor(color)}
                      className={`w-8 h-8 rounded border-2 ${
                        selectedColor === color ? 'border-gray-800' : 'border-gray-300'
                      }`}
                      style={{
                        backgroundColor: {
                          red: '#ef4444',
                          green: '#22c55e',
                          blue: '#3b82f6',
                          yellow: '#eab308',
                          purple: '#a855f7',
                          grey: '#6b7280'
                        }[color]
                      }}
                    />
                  ))}
                </div>
              </div>
            </div>

            {/* Environment Config */}
            <div className="bg-white rounded-lg shadow-lg p-6">
              <h3 className="text-lg font-semibold mb-4">Configuration</h3>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-1">Name</label>
                  <input
                    type="text"
                    value={environmentConfig.name}
                    onChange={(e) => setEnvironmentConfig(prev => ({
                      ...prev, name: e.target.value
                    }))}
                    className="w-full p-2 border rounded-lg"
                  />
                </div>
                
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <label className="block text-sm font-medium mb-1">Width</label>
                    <input
                      type="number"
                      min="5"
                      max="20"
                      value={environmentConfig.width}
                      onChange={(e) => setEnvironmentConfig(prev => ({
                        ...prev, width: parseInt(e.target.value)
                      }))}
                      className="w-full p-2 border rounded-lg"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-1">Height</label>
                    <input
                      type="number"
                      min="5"
                      max="20"
                      value={environmentConfig.height}
                      onChange={(e) => setEnvironmentConfig(prev => ({
                        ...prev, height: parseInt(e.target.value)
                      }))}
                      className="w-full p-2 border rounded-lg"
                    />
                  </div>
                </div>
                
                <div>
                  <label className="block text-sm font-medium mb-1">Agents</label>
                  <input
                    type="number"
                    min="2"
                    max="8"
                    value={environmentConfig.n_agents}
                    onChange={(e) => setEnvironmentConfig(prev => ({
                      ...prev, n_agents: parseInt(e.target.value)
                    }))}
                    className="w-full p-2 border rounded-lg"
                  />
                </div>
              </div>
            </div>

            {/* Templates */}
            <div className="bg-white rounded-lg shadow-lg p-6">
              <h3 className="text-lg font-semibold mb-4">Templates</h3>
              
              {Object.entries(templates).map(([category, templateList]) => (
                <div key={category} className="mb-4">
                  <h4 className="font-medium text-gray-700 mb-2">{category}</h4>
                  <div className="space-y-1">
                    {templateList.map(template => (
                      <button
                        key={template.id}
                        onClick={() => loadTemplate(template)}
                        className="w-full text-left p-2 rounded hover:bg-gray-100 text-sm"
                      >
                        {template.name}
                      </button>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Center Panel - Grid Canvas */}
          <div className="lg:col-span-2">
            <div className="bg-white rounded-lg shadow-lg p-6">
              <h3 className="text-lg font-semibold mb-4 flex items-center">
                <Grid className="w-5 h-5 mr-2" />
                Grid Canvas ({environmentConfig.width}×{environmentConfig.height})
              </h3>
              
              <div className="flex justify-center">
                <div 
                  className="inline-grid gap-1 p-4 bg-gray-100 rounded-lg"
                  style={{
                    gridTemplateColumns: `repeat(${environmentConfig.width}, minmax(0, 1fr))`
                  }}
                >
                  {grid.map((row, y) => 
                    row.map((cell, x) => {
                      const display = getCellDisplay(cell, x, y);
                      return (
                        <button
                          key={`${x}-${y}`}
                          onClick={() => handleCellClick(x, y)}
                          className={`w-8 h-8 border border-gray-300 text-xs flex items-center justify-center transition-all hover:scale-110 ${display.className}`}
                          title={`(${x}, ${y})`}
                        >
                          {display.symbol}
                        </button>
                      );
                    })
                  )}
                </div>
              </div>
              
              <div className="mt-4 text-center text-sm text-gray-600">
                Selected Tool: <span className="font-medium">{selectedTool}</span> | 
                Color: <span className="font-medium">{selectedColor}</span>
              </div>
            </div>
          </div>

          {/* Right Panel - Actions and Preview */}
          <div className="lg:col-span-1 space-y-6">
            
            {/* Actions */}
            <div className="bg-white rounded-lg shadow-lg p-6">
              <h3 className="text-lg font-semibold mb-4">Actions</h3>
              
              <div className="space-y-3">
                <button
                  onClick={saveEnvironment}
                  disabled={saving}
                  className="w-full bg-blue-500 hover:bg-blue-600 text-white p-3 rounded-lg flex items-center justify-center disabled:opacity-50"
                >
                  <Save className="w-4 h-4 mr-2" />
                  {saving ? 'Saving...' : 'Save Environment'}
                </button>
                
                <div className="grid grid-cols-2 gap-2">
                  <button
                    onClick={() => exportEnvironment('yaml')}
                    className="bg-green-500 hover:bg-green-600 text-white p-2 rounded-lg flex items-center justify-center text-sm"
                  >
                    <Download className="w-4 h-4 mr-1" />
                    YAML
                  </button>
                  <button
                    onClick={() => exportEnvironment('python')}
                    className="bg-purple-500 hover:bg-purple-600 text-white p-2 rounded-lg flex items-center justify-center text-sm"
                  >
                    <Download className="w-4 h-4 mr-1" />
                    Python
                  </button>
                </div>
              </div>
            </div>

            {/* Preview */}
            {preview && (
              <div className="bg-white rounded-lg shadow-lg p-6">
                <h3 className="text-lg font-semibold mb-4 flex items-center">
                  <Eye className="w-5 h-5 mr-2" />
                  Preview
                </h3>
                <pre className="bg-gray-900 text-green-400 p-4 rounded-lg text-xs overflow-auto font-mono">
                  {preview}
                </pre>
              </div>
            )}

            {/* Instructions */}
            <div className="bg-white rounded-lg shadow-lg p-6">
              <h3 className="text-lg font-semibold mb-4">Instructions</h3>
              <div className="text-sm space-y-2 text-gray-600">
                <p>• Select a tool from the left panel</p>
                <p>• Choose a color for objects</p>
                <p>• Click on grid cells to place objects</p>
                <p>• Use eraser to remove objects</p>
                <p>• Border walls cannot be modified</p>
                <p>• Save and export when finished</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default EnvironmentBuilder;
