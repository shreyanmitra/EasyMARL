import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { ChevronDownIcon, ChevronUpIcon } from '@heroicons/react/24/outline';
import { trainingAPI } from '../services/api';

/**
 * Algorithms Page Component
 * 
 * Displays comprehensive information about all 21+ MARL algorithms,
 * replicating the algorithm descriptions from the Python GUI.
 * Provides educational content for algorithm selection and understanding.
 */
const AlgorithmsPage = () => {
  const [algorithms, setAlgorithms] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [expandedAlgorithm, setExpandedAlgorithm] = useState(null);

  // Load algorithms from backend API
  useEffect(() => {
    const loadAlgorithms = async () => {
      try {
        const data = await trainingAPI.getAlgorithms();
        setAlgorithms(data);
      } catch (error) {
        console.error('Failed to load algorithms:', error);
        // Fallback to hardcoded data if API fails
        setAlgorithms(getFallbackAlgorithms());
      } finally {
        setLoading(false);
      }
    };

    loadAlgorithms();
  }, []);

  // Get unique categories
  const categories = ['all', ...new Set(algorithms.map(alg => alg.category))];

  // Filter algorithms by category
  const filteredAlgorithms = selectedCategory === 'all' 
    ? algorithms 
    : algorithms.filter(alg => alg.category === selectedCategory);

  // Get difficulty color
  const getDifficultyColor = (difficulty) => {
    switch (difficulty?.toLowerCase()) {
      case 'beginner': return 'bg-green-100 text-green-800';
      case 'intermediate': return 'bg-yellow-100 text-yellow-800';
      case 'advanced': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  // Get category color
  const getCategoryColor = (category) => {
    switch (category) {
      case 'Value Decomposition': return 'bg-blue-100 text-blue-800';
      case 'Actor-Critic': return 'bg-purple-100 text-purple-800';
      case 'Game-Theoretic': return 'bg-orange-100 text-orange-800';
      case 'Large-Scale': return 'bg-green-100 text-green-800';
      case 'Communication': return 'bg-pink-100 text-pink-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading algorithms...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto space-y-8">
      {/* Page Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center"
      >
        <h1 className="text-4xl font-bold text-gray-900 mb-4">
          🧠 MARL Algorithm Catalog
        </h1>
        <p className="text-xl text-gray-600 mb-8">
          Explore 21+ state-of-the-art multi-agent reinforcement learning algorithms
        </p>

        {/* Category Filter */}
        <div className="flex flex-wrap justify-center gap-2">
          {categories.map(category => (
            <button
              key={category}
              onClick={() => setSelectedCategory(category)}
              className={`px-4 py-2 rounded-full text-sm font-medium transition-colors ${
                selectedCategory === category
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
              }`}
            >
              {category === 'all' ? 'All Categories' : category}
            </button>
          ))}
        </div>
      </motion.div>

      {/* Algorithm Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {filteredAlgorithms.map((algorithm, index) => (
          <motion.div
            key={algorithm.value}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
            className="bg-white rounded-xl shadow-lg overflow-hidden hover:shadow-xl transition-shadow"
          >
            {/* Algorithm Header */}
            <div className="p-6 border-b border-gray-200">
              <div className="flex items-start justify-between mb-4">
                <h3 className="text-2xl font-bold text-gray-900">
                  {algorithm.label}
                </h3>
                <span className={`px-2 py-1 rounded-full text-xs font-medium ${getDifficultyColor(algorithm.difficulty)}`}>
                  {algorithm.difficulty || 'Intermediate'}
                </span>
              </div>

              <span className={`inline-block px-3 py-1 rounded-full text-sm font-medium ${getCategoryColor(algorithm.category)}`}>
                {algorithm.category}
              </span>

              <p className="text-gray-600 mt-4 line-clamp-3">
                {algorithm.description}
              </p>
            </div>

            {/* Algorithm Details */}
            <div className="p-6">
              <button
                onClick={() => setExpandedAlgorithm(
                  expandedAlgorithm === algorithm.value ? null : algorithm.value
                )}
                className="w-full flex items-center justify-between text-blue-600 font-medium hover:text-blue-800 transition-colors"
              >
                <span>View Details</span>
                {expandedAlgorithm === algorithm.value ? (
                  <ChevronUpIcon className="h-5 w-5" />
                ) : (
                  <ChevronDownIcon className="h-5 w-5" />
                )}
              </button>

              {expandedAlgorithm === algorithm.value && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  className="mt-4 space-y-4"
                >
                  {/* Use Case */}
                  {algorithm.use_case && (
                    <div>
                      <h4 className="font-semibold text-gray-900 mb-2">🎯 Best For:</h4>
                      <p className="text-gray-600 text-sm">{algorithm.use_case}</p>
                    </div>
                  )}

                  {/* Pros */}
                  {algorithm.pros && algorithm.pros.length > 0 && (
                    <div>
                      <h4 className="font-semibold text-gray-900 mb-2 text-green-700">✅ Advantages:</h4>
                      <ul className="text-sm text-gray-600 space-y-1">
                        {algorithm.pros.map((pro, idx) => (
                          <li key={idx} className="flex items-start">
                            <span className="text-green-500 mr-2">•</span>
                            {pro}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {/* Cons */}
                  {algorithm.cons && algorithm.cons.length > 0 && (
                    <div>
                      <h4 className="font-semibold text-gray-900 mb-2 text-red-700">⚠️ Limitations:</h4>
                      <ul className="text-sm text-gray-600 space-y-1">
                        {algorithm.cons.map((con, idx) => (
                          <li key={idx} className="flex items-start">
                            <span className="text-red-500 mr-2">•</span>
                            {con}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {/* Try Algorithm Button */}
                  <div className="pt-4 border-t border-gray-200">
                    <a
                      href={`/training?algorithm=${algorithm.value}`}
                      className="w-full bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded-lg text-center block transition-colors"
                    >
                      🚀 Try {algorithm.label}
                    </a>
                  </div>
                </motion.div>
              )}
            </div>
          </motion.div>
        ))}
      </div>

      {/* No Results */}
      {filteredAlgorithms.length === 0 && (
        <div className="text-center py-12">
          <div className="text-6xl mb-4">🔍</div>
          <h3 className="text-xl font-semibold text-gray-700 mb-2">
            No algorithms found
          </h3>
          <p className="text-gray-500">
            Try selecting a different category or check back later.
          </p>
        </div>
      )}

      {/* Learning Path Recommendations */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
        className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl p-8 mt-12"
      >
        <h2 className="text-2xl font-bold text-gray-900 mb-6">
          🎓 Recommended Learning Path
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="text-center">
            <div className="text-4xl mb-4">🟢</div>
            <h3 className="text-lg font-semibold text-green-800 mb-2">Beginner</h3>
            <p className="text-sm text-gray-600 mb-4">
              Start with simple, well-understood algorithms
            </p>
            <div className="space-y-2">
              <div className="bg-white px-3 py-2 rounded-lg text-sm">QMIX</div>
              <div className="bg-white px-3 py-2 rounded-lg text-sm">VDN</div>
              <div className="bg-white px-3 py-2 rounded-lg text-sm">IPPO</div>
            </div>
          </div>

          <div className="text-center">
            <div className="text-4xl mb-4">🟡</div>
            <h3 className="text-lg font-semibold text-yellow-800 mb-2">Intermediate</h3>
            <p className="text-sm text-gray-600 mb-4">
              Explore more sophisticated approaches
            </p>
            <div className="space-y-2">
              <div className="bg-white px-3 py-2 rounded-lg text-sm">MAPPO</div>
              <div className="bg-white px-3 py-2 rounded-lg text-sm">MADDPG</div>
              <div className="bg-white px-3 py-2 rounded-lg text-sm">COMA</div>
            </div>
          </div>

          <div className="text-center">
            <div className="text-4xl mb-4">🔴</div>
            <h3 className="text-lg font-semibold text-red-800 mb-2">Advanced</h3>
            <p className="text-sm text-gray-600 mb-4">
              Cutting-edge research algorithms
            </p>
            <div className="space-y-2">
              <div className="bg-white px-3 py-2 rounded-lg text-sm">QTRAN</div>
              <div className="bg-white px-3 py-2 rounded-lg text-sm">MAVEN</div>
              <div className="bg-white px-3 py-2 rounded-lg text-sm">DCG</div>
            </div>
          </div>
        </div>
      </motion.div>
    </div>
  );
};

/**
 * Fallback algorithm data if API is not available
 * This ensures the page works even without backend connectivity
 */
const getFallbackAlgorithms = () => [
  {
    value: 'qmix',
    label: 'QMIX',
    category: 'Value Decomposition',
    difficulty: 'Beginner',
    description: 'Monotonic value function factorization for cooperative multi-agent tasks',
    use_case: 'Cooperative tasks where team reward needs to be decomposed',
    pros: ['Easy to understand', 'Good performance on cooperative tasks', 'Stable training'],
    cons: ['Limited to monotonic value functions', 'May not work well with conflicting objectives']
  },
  {
    value: 'mappo',
    label: 'MAPPO',
    category: 'Actor-Critic',
    difficulty: 'Intermediate',
    description: 'Multi-Agent Proximal Policy Optimization with centralized training',
    use_case: 'General purpose MARL with both cooperative and competitive elements',
    pros: ['Reliable performance', 'Good sample efficiency', 'Works across many domains'],
    cons: ['Can be complex to tune', 'Requires careful hyperparameter selection']
  },
  // Add more fallback algorithms as needed...
];

export default AlgorithmsPage;
