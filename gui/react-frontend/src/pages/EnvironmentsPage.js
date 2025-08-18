import React, { useState } from 'react';
import { 
  MapIcon, 
  ChipIcon, 
  PlayIcon,
  InformationCircleIcon 
} from '@heroicons/react/outline';

/**
 * EnvironmentsPage Component - Browse Available MARL Environments
 * 
 * This page showcases all available multi-agent environments in EasyMARL,
 * providing detailed information about each environment and allowing users
 * to preview and understand different training scenarios.
 */
function EnvironmentsPage() {
  const [selectedCategory, setSelectedCategory] = useState('all');

  const environments = [
    {
      name: "MultiGrid-Empty-8x8",
      category: "navigation",
      description: "Simple empty grid where agents learn basic navigation and coordination",
      agents: "2-4",
      difficulty: "Beginner",
      features: ["Navigation", "Coordination", "Multi-agent"],
      image: "/placeholder-env.png"
    },
    {
      name: "MultiGrid-Cluttered-15x15",
      category: "navigation", 
      description: "Complex grid with obstacles requiring advanced pathfinding and coordination",
      agents: "2-6",
      difficulty: "Intermediate",
      features: ["Pathfinding", "Obstacle Avoidance", "Team Coordination"],
      image: "/placeholder-env.png"
    },
    {
      name: "MultiGrid-DoorKey-8x8",
      category: "puzzle",
      description: "Agents must find keys to unlock doors and reach goals cooperatively",
      agents: "2-3",
      difficulty: "Intermediate",
      features: ["Puzzle Solving", "Sequential Tasks", "Cooperation"],
      image: "/placeholder-env.png"
    },
    {
      name: "MultiGrid-FourRooms-15x15",
      category: "navigation",
      description: "Classic four rooms environment adapted for multi-agent scenarios",
      agents: "2-4",
      difficulty: "Intermediate",
      features: ["Exploration", "Room Navigation", "Coordination"],
      image: "/placeholder-env.png"
    },
    {
      name: "MultiGrid-Collect-10x10",
      category: "collection",
      description: "Agents compete or cooperate to collect items scattered in the environment",
      agents: "2-8",
      difficulty: "Beginner",
      features: ["Resource Collection", "Competition", "Cooperation"],
      image: "/placeholder-env.png"
    },
    {
      name: "MultiGrid-CoinGame-8x8",
      category: "game",
      description: "Strategic coin collection game with competitive and cooperative elements",
      agents: "2",
      difficulty: "Advanced",
      features: ["Strategic Thinking", "Game Theory", "Competition"],
      image: "/placeholder-env.png"
    }
  ];

  const categories = [
    { id: 'all', name: 'All Environments', count: environments.length },
    { id: 'navigation', name: 'Navigation', count: environments.filter(e => e.category === 'navigation').length },
    { id: 'puzzle', name: 'Puzzle', count: environments.filter(e => e.category === 'puzzle').length },
    { id: 'collection', name: 'Collection', count: environments.filter(e => e.category === 'collection').length },
    { id: 'game', name: 'Game Theory', count: environments.filter(e => e.category === 'game').length }
  ];

  const filteredEnvironments = selectedCategory === 'all' 
    ? environments 
    : environments.filter(env => env.category === selectedCategory);

  const getDifficultyColor = (difficulty) => {
    switch (difficulty) {
      case 'Beginner': return 'bg-green-100 text-green-800';
      case 'Intermediate': return 'bg-yellow-100 text-yellow-800';
      case 'Advanced': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="text-center">
        <MapIcon className="h-12 w-12 text-blue-600 mx-auto mb-4" />
        <h1 className="text-4xl font-bold text-gray-900 mb-4">
          MARL Environments
        </h1>
        <p className="text-xl text-gray-600 max-w-3xl mx-auto">
          Explore our collection of multi-agent environments designed for learning and research.
          Each environment offers unique challenges and learning opportunities.
        </p>
      </div>

      {/* Category Filter */}
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">
          Filter by Category
        </h2>
        <div className="flex flex-wrap gap-3">
          {categories.map((category) => (
            <button
              key={category.id}
              onClick={() => setSelectedCategory(category.id)}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                selectedCategory === category.id
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              {category.name} ({category.count})
            </button>
          ))}
        </div>
      </div>

      {/* Environments Grid */}
      <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
        {filteredEnvironments.map((env, index) => (
          <div key={index} className="bg-white rounded-lg shadow-lg overflow-hidden hover:shadow-xl transition-shadow">
            {/* Environment Preview */}
            <div className="h-48 bg-gradient-to-br from-blue-100 to-purple-100 flex items-center justify-center">
              <div className="text-center">
                <ChipIcon className="h-16 w-16 text-blue-600 mx-auto mb-2" />
                <p className="text-sm text-gray-600">Environment Preview</p>
              </div>
            </div>

            {/* Environment Info */}
            <div className="p-6">
              <div className="flex items-start justify-between mb-3">
                <h3 className="text-lg font-semibold text-gray-900">
                  {env.name}
                </h3>
                <span className={`px-2 py-1 rounded-full text-xs font-medium ${getDifficultyColor(env.difficulty)}`}>
                  {env.difficulty}
                </span>
              </div>

              <p className="text-gray-600 text-sm mb-4">
                {env.description}
              </p>

              {/* Environment Stats */}
              <div className="grid grid-cols-2 gap-4 mb-4 text-sm">
                <div>
                  <span className="font-medium text-gray-700">Agents:</span>
                  <span className="ml-2 text-gray-600">{env.agents}</span>
                </div>
                <div>
                  <span className="font-medium text-gray-700">Category:</span>
                  <span className="ml-2 text-gray-600 capitalize">{env.category}</span>
                </div>
              </div>

              {/* Features */}
              <div className="mb-4">
                <h4 className="text-sm font-medium text-gray-700 mb-2">Features:</h4>
                <div className="flex flex-wrap gap-2">
                  {env.features.map((feature, idx) => (
                    <span 
                      key={idx}
                      className="px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded-full"
                    >
                      {feature}
                    </span>
                  ))}
                </div>
              </div>

              {/* Action Buttons */}
              <div className="flex space-x-2">
                <button className="flex-1 flex items-center justify-center px-3 py-2 bg-blue-600 text-white text-sm font-medium rounded-md hover:bg-blue-700 transition-colors">
                  <PlayIcon className="h-4 w-4 mr-1" />
                  Train Here
                </button>
                <button className="px-3 py-2 border border-gray-300 text-gray-700 text-sm font-medium rounded-md hover:bg-gray-50 transition-colors">
                  <InformationCircleIcon className="h-4 w-4" />
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Environment Creation Guide */}
      <div className="bg-gradient-to-r from-purple-600 to-blue-600 rounded-lg p-8 text-white">
        <h2 className="text-2xl font-bold mb-4">
          Want to Create Your Own Environment?
        </h2>
        <p className="text-purple-100 mb-6">
          EasyMARL supports custom environment creation. Follow our comprehensive guide 
          to build environments tailored to your research needs.
        </p>
        <div className="space-x-4">
          <button className="px-6 py-3 bg-white text-purple-600 font-medium rounded-md hover:bg-gray-100 transition-colors">
            Environment Creation Guide
          </button>
          <button className="px-6 py-3 border border-white text-white font-medium rounded-md hover:bg-white hover:text-purple-600 transition-colors">
            View Examples
          </button>
        </div>
      </div>

      {/* Technical Details */}
      <div className="bg-white rounded-lg shadow-lg p-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-6">
          Environment Technical Specifications
        </h2>
        <div className="grid md:grid-cols-2 gap-8">
          <div>
            <h3 className="text-lg font-semibold text-gray-900 mb-3">
              Observation Space
            </h3>
            <ul className="space-y-2 text-gray-600">
              <li>• Grid-based visual observations</li>
              <li>• Agent position and orientation</li>
              <li>• Object and obstacle locations</li>
              <li>• Goal and task-specific information</li>
            </ul>
          </div>
          <div>
            <h3 className="text-lg font-semibold text-gray-900 mb-3">
              Action Space
            </h3>
            <ul className="space-y-2 text-gray-600">
              <li>• Discrete movement actions (up, down, left, right)</li>
              <li>• Object interaction (pick up, drop, toggle)</li>
              <li>• Communication actions (where applicable)</li>
              <li>• Environment-specific actions</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}

export default EnvironmentsPage;
