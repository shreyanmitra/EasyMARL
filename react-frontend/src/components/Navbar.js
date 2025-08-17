import React from 'react';
import { Link, useLocation } from 'react-router-dom';

/**
 * Navigation Bar Component
 * 
 * Provides navigation between different pages of the EasyMARL interface
 * with visual indicators for the current page and research-oriented features.
 */
const Navbar = () => {
  const location = useLocation();

  const navItems = [
    { path: '/', label: '🏠 Home', desc: 'Overview' },
    { path: '/training', label: '🚀 Training', desc: 'Start Training' },
    { path: '/manual-control', label: '🎮 Manual Control', desc: 'Interactive Control' },
    { path: '/algorithms', label: '🧠 Algorithms', desc: 'Browse Algorithms' },
    { path: '/research', label: '🔬 Research', desc: 'Advanced Tools' },
    { path: '/tutorial', label: '📚 Tutorial', desc: 'Learn MARL' },
    { path: '/help', label: '❓ Help', desc: 'Documentation' }
  ];

  return (
    <nav className="bg-white shadow-lg border-b border-gray-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          {/* Logo and Brand */}
          <div className="flex items-center">
            <Link to="/" className="flex items-center space-x-3">
              <div className="flex items-center justify-center w-10 h-10 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg">
                <span className="text-white font-bold text-xl">E</span>
              </div>
              <div>
                <h1 className="text-xl font-bold text-gray-900">EasyMARL</h1>
                <p className="text-xs text-gray-500">Multi-Agent RL Framework</p>
              </div>
            </Link>
          </div>

          {/* Navigation Links */}
          <div className="hidden md:flex items-center space-x-1">
            {navItems.map((item) => {
              const isActive = location.pathname === item.path;
              return (
                <Link
                  key={item.path}
                  to={item.path}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors duration-200 ${
                    isActive
                      ? 'bg-blue-100 text-blue-700 border border-blue-200'
                      : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
                  }`}
                >
                  <div className="text-center">
                    <div>{item.label}</div>
                    <div className="text-xs text-gray-400">{item.desc}</div>
                  </div>
                </Link>
              );
            })}
          </div>

          {/* Mobile Menu Button */}
          <div className="md:hidden flex items-center">
            <button className="text-gray-600 hover:text-gray-900 focus:outline-none focus:text-gray-900">
              <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            </button>
          </div>
        </div>
      </div>

      {/* Mobile Navigation (hidden by default) */}
      <div className="md:hidden">
        <div className="px-2 pt-2 pb-3 space-y-1 sm:px-3 bg-gray-50">
          {navItems.map((item) => {
            const isActive = location.pathname === item.path;
            return (
              <Link
                key={item.path}
                to={item.path}
                className={`block px-3 py-2 rounded-md text-base font-medium ${
                  isActive
                    ? 'bg-blue-100 text-blue-700'
                    : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
                }`}
              >
                {item.label}
              </Link>
            );
          })}
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
