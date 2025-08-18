import React from 'react';
import { Link } from 'react-router-dom';
import { 
  PlayIcon, 
  BookOpenIcon, 
  BeakerIcon,
  ChipIcon,
  ChartBarIcon,
  AcademicCapIcon,
  PlayIcon as PlayCircleIcon  // Add PlayCircleIcon as alias
} from '@heroicons/react/outline';

/**
 * HomePage Component - Welcome page for EasyMARL
 * 
 * This is the landing page that introduces users to EasyMARL and provides
 * quick navigation to key features. It's designed to be welcoming for
 * beginners while highlighting advanced capabilities for researchers.
 */
function HomePage() {
  const features = [
    {
      title: "Interactive Training",
      description: "Train multi-agent systems with real-time visualization and monitoring",
      icon: PlayCircleIcon,
      link: "/training",
      color: "bg-blue-500"
    },
    {
      title: "Learn MARL",
      description: "Comprehensive tutorials covering multi-agent reinforcement learning fundamentals",
      icon: AcademicCapIcon,
      link: "/tutorial",
      color: "bg-green-500"
    },
    {
      title: "Algorithm Explorer",
      description: "Discover and compare 21+ state-of-the-art MARL algorithms",
      icon: BeakerIcon,
      link: "/algorithms",
      color: "bg-purple-500"
    },
    {
      title: "Research Tools",
      description: "Advanced experimentation and analysis tools for MARL research",
      icon: ChartBarIcon,
      link: "/research",
      color: "bg-orange-500"
    }
  ];

  const stats = [
    { label: "MARL Algorithms", value: "21+" },
    { label: "Environment Types", value: "10+" },
    { label: "Research Papers", value: "50+" },
    { label: "Active Users", value: "1000+" }
  ];

  return (
    <div className="space-y-12">
      {/* Hero Section */}
      <div className="text-center py-12">
        <div className="mb-8">
          <ChipIcon className="h-16 w-16 text-blue-600 mx-auto mb-4" />
          <h1 className="text-5xl font-bold text-gray-900 mb-4">
            Welcome to EasyMARL
          </h1>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            The most comprehensive and user-friendly platform for 
            <span className="text-blue-600 font-semibold"> Multi-Agent Reinforcement Learning</span>.
            From beginners to researchers, EasyMARL makes MARL accessible to everyone.
          </p>
        </div>
        
        <div className="space-x-4">
          <Link 
            to="/training" 
            className="inline-flex items-center px-6 py-3 border border-transparent text-base font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 transition-colors"
          >
            <PlayIcon className="h-5 w-5 mr-2" />
            Start Training
          </Link>
          <Link 
            to="/tutorial" 
            className="inline-flex items-center px-6 py-3 border border-gray-300 text-base font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 transition-colors"
          >
            <BookOpenIcon className="h-5 w-5 mr-2" />
            Learn MARL
          </Link>
        </div>
      </div>

      {/* Stats Section */}
      <div className="bg-white rounded-lg shadow-lg p-8">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
          {stats.map((stat, index) => (
            <div key={index} className="text-center">
              <div className="text-3xl font-bold text-blue-600 mb-2">
                {stat.value}
              </div>
              <div className="text-gray-600">
                {stat.label}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Features Grid */}
      <div className="grid md:grid-cols-2 gap-8">
        {features.map((feature, index) => {
          const IconComponent = feature.icon;
          return (
            <Link
              key={index}
              to={feature.link}
              className="block p-6 bg-white rounded-lg shadow-lg hover:shadow-xl transition-shadow group"
            >
              <div className="flex items-start space-x-4">
                <div className={`p-3 rounded-lg ${feature.color}`}>
                  <IconComponent className="h-6 w-6 text-white" />
                </div>
                <div className="flex-1">
                  <h3 className="text-xl font-semibold text-gray-900 group-hover:text-blue-600 transition-colors">
                    {feature.title}
                  </h3>
                  <p className="text-gray-600 mt-2">
                    {feature.description}
                  </p>
                </div>
              </div>
            </Link>
          );
        })}
      </div>

      {/* What Makes EasyMARL Special */}
      <div className="bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg p-8 text-white">
        <h2 className="text-3xl font-bold mb-6 text-center">
          What Makes EasyMARL Special?
        </h2>
        <div className="grid md:grid-cols-3 gap-8">
          <div className="text-center">
            <h3 className="text-xl font-semibold mb-3">🎯 Beginner Friendly</h3>
            <p className="text-blue-100">
              No command-line experience required. Point-and-click interface 
              with comprehensive tutorials and guided learning paths.
            </p>
          </div>
          <div className="text-center">
            <h3 className="text-xl font-semibold mb-3">🚀 Research Ready</h3>
            <p className="text-blue-100">
              Advanced features for researchers including experiment tracking,
              custom environments, and publication-ready results.
            </p>
          </div>
          <div className="text-center">
            <h3 className="text-xl font-semibold mb-3">⚡ High Performance</h3>
            <p className="text-blue-100">
              10x faster training with enhanced vectorization, JIT compilation,
              and world-class optimizations.
            </p>
          </div>
        </div>
      </div>

      {/* Getting Started */}
      <div className="bg-white rounded-lg shadow-lg p-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-6 text-center">
          Ready to Get Started?
        </h2>
        <div className="grid md:grid-cols-3 gap-6">
          <div className="text-center p-4 border border-gray-200 rounded-lg">
            <div className="text-2xl font-bold text-blue-600 mb-2">1</div>
            <h3 className="font-semibold mb-2">Choose Your Path</h3>
            <p className="text-gray-600 text-sm">
              Start with tutorials if you're new to MARL, or jump straight to training if you're experienced.
            </p>
          </div>
          <div className="text-center p-4 border border-gray-200 rounded-lg">
            <div className="text-2xl font-bold text-blue-600 mb-2">2</div>
            <h3 className="font-semibold mb-2">Select Algorithm & Environment</h3>
            <p className="text-gray-600 text-sm">
              Browse our collection of algorithms and environments to find the perfect combination for your research.
            </p>
          </div>
          <div className="text-center p-4 border border-gray-200 rounded-lg">
            <div className="text-2xl font-bold text-blue-600 mb-2">3</div>
            <h3 className="font-semibold mb-2">Train & Analyze</h3>
            <p className="text-gray-600 text-sm">
              Watch your agents learn in real-time and analyze their performance with comprehensive visualizations.
            </p>
          </div>
        </div>
        
        <div className="text-center mt-8">
          <Link 
            to="/tutorial" 
            className="inline-flex items-center px-8 py-3 border border-transparent text-lg font-medium rounded-md text-white bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 transition-all"
          >
            Begin Your MARL Journey
          </Link>
        </div>
      </div>
    </div>
  );
}

export default HomePage;
