import React from 'react';
import { Github, Twitter, Mail, ExternalLink } from 'lucide-react';

const Footer = () => {
  return (
    <footer className="bg-gray-900 text-white py-12">
      <div className="max-w-7xl mx-auto px-4">
        <div className="grid md:grid-cols-4 gap-8">
          {/* About */}
          <div className="md:col-span-2">
            <h3 className="text-2xl font-bold mb-4 gradient-text">EasyMARL</h3>
            <p className="text-gray-300 mb-4 max-w-md">
              Comprehensive Multi-Agent Reinforcement Learning framework designed for 
              education and research. Making MARL accessible to everyone.
            </p>
            <div className="flex space-x-4">
              <a
                href="https://github.com/shreyanmitra/EasyMARL"
                target="_blank"
                rel="noopener noreferrer"
                className="text-gray-400 hover:text-white transition-colors duration-200"
              >
                <Github className="w-6 h-6" />
              </a>
              <a
                href="mailto:shreyan.m.mitra@gmail.com"
                className="text-gray-400 hover:text-white transition-colors duration-200"
              >
                <Mail className="w-6 h-6" />
              </a>
            </div>
          </div>

          {/* Quick Links */}
          <div>
            <h4 className="text-lg font-semibold mb-4">Quick Links</h4>
            <ul className="space-y-2">
              <li>
                <a href="/docs" className="text-gray-300 hover:text-white transition-colors duration-200">
                  Documentation
                </a>
              </li>
              <li>
                <a href="/algorithms" className="text-gray-300 hover:text-white transition-colors duration-200">
                  Algorithms
                </a>
              </li>
              <li>
                <a href="/tutorials" className="text-gray-300 hover:text-white transition-colors duration-200">
                  Tutorials
                </a>
              </li>
              <li>
                <a href="/examples" className="text-gray-300 hover:text-white transition-colors duration-200">
                  Examples
                </a>
              </li>
            </ul>
          </div>

          {/* Resources */}
          <div>
            <h4 className="text-lg font-semibold mb-4">Resources</h4>
            <ul className="space-y-2">
              <li>
                <a 
                  href="https://github.com/shreyanmitra/EasyMARL"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center space-x-2 text-gray-300 hover:text-white transition-colors duration-200"
                >
                  <span>GitHub Repository</span>
                  <ExternalLink className="w-4 h-4" />
                </a>
              </li>
              <li>
                <a 
                  href="https://github.com/shreyanmitra/EasyMARL/issues"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center space-x-2 text-gray-300 hover:text-white transition-colors duration-200"
                >
                  <span>Issue Tracker</span>
                  <ExternalLink className="w-4 h-4" />
                </a>
              </li>
              <li>
                <a 
                  href="https://pypi.org/project/easymarl/"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center space-x-2 text-gray-300 hover:text-white transition-colors duration-200"
                >
                  <span>PyPI Package</span>
                  <ExternalLink className="w-4 h-4" />
                </a>
              </li>
            </ul>
          </div>
        </div>

        <div className="border-t border-gray-800 mt-8 pt-8 text-center">
          <p className="text-gray-400">
            © 2025 EasyMARL. Built with ❤️ by{' '}
            <a 
              href="https://github.com/shreyanmitra"
              target="_blank"
              rel="noopener noreferrer"
              className="text-blue-400 hover:text-blue-300 transition-colors duration-200"
            >
              Shreyan Mitra
            </a>
            . Licensed under MIT.
          </p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
