import React, { useState } from 'react';
import { 
  QuestionMarkCircleIcon, 
  BookOpenIcon,
  ChatIcon,
  DocumentTextIcon,
  VideoCameraIcon,
  CodeIcon,
  AcademicCapIcon,
  ExclamationIcon
} from '@heroicons/react/outline';

/**
 * HelpPage Component - Comprehensive Help and Documentation
 * 
 * This page provides users with comprehensive help resources including
 * documentation, tutorials, troubleshooting, and community support
 * for the EasyMARL platform.
 */
function HelpPage() {
  const [activeTab, setActiveTab] = useState('getting-started');

  const helpSections = [
    {
      id: 'getting-started',
      title: 'Getting Started',
      icon: AcademicCapIcon,
      content: (
        <div className="space-y-6">
          <h3 className="text-xl font-semibold text-gray-900">Welcome to EasyMARL!</h3>
          <p className="text-gray-600">
            This guide will help you get started with multi-agent reinforcement learning using EasyMARL.
          </p>
          
          <div className="space-y-4">
            <div className="border border-gray-200 rounded-lg p-4">
              <h4 className="font-semibold text-gray-900 mb-2">Step 1: Choose Your Learning Path</h4>
              <p className="text-gray-600 text-sm">
                If you're new to MARL, start with our <a href="/tutorial" className="text-blue-600 hover:underline">comprehensive tutorial</a>. 
                If you're experienced, jump directly to <a href="/training" className="text-blue-600 hover:underline">training</a>.
              </p>
            </div>
            
            <div className="border border-gray-200 rounded-lg p-4">
              <h4 className="font-semibold text-gray-900 mb-2">Step 2: Select Algorithm and Environment</h4>
              <p className="text-gray-600 text-sm">
                Browse our collection of <a href="/algorithms" className="text-blue-600 hover:underline">21+ algorithms</a> and 
                choose from various <a href="/environments" className="text-blue-600 hover:underline">multi-agent environments</a>.
              </p>
            </div>
            
            <div className="border border-gray-200 rounded-lg p-4">
              <h4 className="font-semibold text-gray-900 mb-2">Step 3: Configure and Train</h4>
              <p className="text-gray-600 text-sm">
                Set your training parameters, enable enhanced features for 10x performance, and watch your agents learn in real-time.
              </p>
            </div>
          </div>
        </div>
      )
    },
    {
      id: 'documentation',
      title: 'Documentation',
      icon: DocumentTextIcon,
      content: (
        <div className="space-y-6">
          <h3 className="text-xl font-semibold text-gray-900">Technical Documentation</h3>
          
          <div className="grid md:grid-cols-2 gap-6">
            <div className="border border-gray-200 rounded-lg p-4">
              <h4 className="font-semibold text-gray-900 mb-2">API Reference</h4>
              <p className="text-gray-600 text-sm mb-3">
                Complete API documentation for all EasyMARL components.
              </p>
              <button className="text-blue-600 hover:underline text-sm">View API Docs →</button>
            </div>
            
            <div className="border border-gray-200 rounded-lg p-4">
              <h4 className="font-semibold text-gray-900 mb-2">Algorithm Guide</h4>
              <p className="text-gray-600 text-sm mb-3">
                Detailed explanations of all implemented MARL algorithms.
              </p>
              <a href="/algorithms" className="text-blue-600 hover:underline text-sm">Explore Algorithms →</a>
            </div>
            
            <div className="border border-gray-200 rounded-lg p-4">
              <h4 className="font-semibold text-gray-900 mb-2">Environment Guide</h4>
              <p className="text-gray-600 text-sm mb-3">
                Learn about available environments and how to create custom ones.
              </p>
              <a href="/environments" className="text-blue-600 hover:underline text-sm">View Environments →</a>
            </div>
            
            <div className="border border-gray-200 rounded-lg p-4">
              <h4 className="font-semibold text-gray-900 mb-2">Research Tools</h4>
              <p className="text-gray-600 text-sm mb-3">
                Advanced features for research and experimentation.
              </p>
              <a href="/research" className="text-blue-600 hover:underline text-sm">Research Features →</a>
            </div>
          </div>
        </div>
      )
    },
    {
      id: 'troubleshooting',
      title: 'Troubleshooting',
      icon: ExclamationIcon,
      content: (
        <div className="space-y-6">
          <h3 className="text-xl font-semibold text-gray-900">Common Issues and Solutions</h3>
          
          <div className="space-y-4">
            <div className="border-l-4 border-yellow-400 bg-yellow-50 p-4">
              <h4 className="font-semibold text-gray-900 mb-2">Training is slow or not working</h4>
              <ul className="text-gray-600 text-sm space-y-1">
                <li>• Check if Enhanced Features are enabled for 10x performance boost</li>
                <li>• Ensure you have sufficient memory (8GB+ recommended)</li>
                <li>• Try reducing the number of parallel environments</li>
                <li>• Check GPU availability for accelerated training</li>
              </ul>
            </div>
            
            <div className="border-l-4 border-blue-400 bg-blue-50 p-4">
              <h4 className="font-semibold text-gray-900 mb-2">Environment not loading</h4>
              <ul className="text-gray-600 text-sm space-y-1">
                <li>• Verify environment name spelling</li>
                <li>• Check if all required dependencies are installed</li>
                <li>• Try refreshing the page</li>
                <li>• Contact support if the issue persists</li>
              </ul>
            </div>
            
            <div className="border-l-4 border-green-400 bg-green-50 p-4">
              <h4 className="font-semibold text-gray-900 mb-2">Algorithm not converging</h4>
              <ul className="text-gray-600 text-sm space-y-1">
                <li>• Try adjusting learning rate (typically 0.0001 - 0.01)</li>
                <li>• Increase training episodes</li>
                <li>• Check if the environment is too complex for the algorithm</li>
                <li>• Consider using curriculum learning</li>
              </ul>
            </div>
            
            <div className="border-l-4 border-red-400 bg-red-50 p-4">
              <h4 className="font-semibold text-gray-900 mb-2">Getting error messages</h4>
              <ul className="text-gray-600 text-sm space-y-1">
                <li>• Check the browser console for detailed error information</li>
                <li>• Ensure you're using a supported browser (Chrome, Firefox, Safari)</li>
                <li>• Clear browser cache and cookies</li>
                <li>• Report the error with full details to our support team</li>
              </ul>
            </div>
          </div>
        </div>
      )
    },
    {
      id: 'tutorials',
      title: 'Video Tutorials',
      icon: VideoCameraIcon,
      content: (
        <div className="space-y-6">
          <h3 className="text-xl font-semibold text-gray-900">Video Learning Resources</h3>
          
          <div className="grid md:grid-cols-2 gap-6">
            <div className="border border-gray-200 rounded-lg p-4">
              <div className="aspect-video bg-gray-100 rounded-lg mb-3 flex items-center justify-center">
                <VideoCameraIcon className="h-12 w-12 text-gray-400" />
              </div>
              <h4 className="font-semibold text-gray-900 mb-2">Getting Started with EasyMARL</h4>
              <p className="text-gray-600 text-sm">A complete walkthrough of the EasyMARL interface and basic training.</p>
            </div>
            
            <div className="border border-gray-200 rounded-lg p-4">
              <div className="aspect-video bg-gray-100 rounded-lg mb-3 flex items-center justify-center">
                <VideoCameraIcon className="h-12 w-12 text-gray-400" />
              </div>
              <h4 className="font-semibold text-gray-900 mb-2">Understanding MARL Algorithms</h4>
              <p className="text-gray-600 text-sm">Deep dive into multi-agent reinforcement learning concepts and algorithms.</p>
            </div>
            
            <div className="border border-gray-200 rounded-lg p-4">
              <div className="aspect-video bg-gray-100 rounded-lg mb-3 flex items-center justify-center">
                <VideoCameraIcon className="h-12 w-12 text-gray-400" />
              </div>
              <h4 className="font-semibold text-gray-900 mb-2">Advanced Research Features</h4>
              <p className="text-gray-600 text-sm">Learn how to use EasyMARL for research and publish-quality experiments.</p>
            </div>
            
            <div className="border border-gray-200 rounded-lg p-4">
              <div className="aspect-video bg-gray-100 rounded-lg mb-3 flex items-center justify-center">
                <VideoCameraIcon className="h-12 w-12 text-gray-400" />
              </div>
              <h4 className="font-semibold text-gray-900 mb-2">Creating Custom Environments</h4>
              <p className="text-gray-600 text-sm">Step-by-step guide to building your own multi-agent environments.</p>
            </div>
          </div>
        </div>
      )
    },
    {
      id: 'community',
      title: 'Community & Support',
      icon: ChatIcon,
      content: (
        <div className="space-y-6">
          <h3 className="text-xl font-semibold text-gray-900">Get Help from the Community</h3>
          
          <div className="grid md:grid-cols-2 gap-6">
            <div className="border border-gray-200 rounded-lg p-6">
              <h4 className="font-semibold text-gray-900 mb-3">Discord Community</h4>
              <p className="text-gray-600 text-sm mb-4">
                Join our active Discord server to chat with other MARL researchers and get real-time help.
              </p>
              <button className="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 transition-colors">
                Join Discord
              </button>
            </div>
            
            <div className="border border-gray-200 rounded-lg p-6">
              <h4 className="font-semibold text-gray-900 mb-3">GitHub Discussions</h4>
              <p className="text-gray-600 text-sm mb-4">
                Browse existing discussions or start a new thread for technical questions and feature requests.
              </p>
              <button className="w-full bg-gray-800 text-white py-2 px-4 rounded-md hover:bg-gray-900 transition-colors">
                View Discussions
              </button>
            </div>
            
            <div className="border border-gray-200 rounded-lg p-6">
              <h4 className="font-semibold text-gray-900 mb-3">Email Support</h4>
              <p className="text-gray-600 text-sm mb-4">
                Contact our support team directly for technical issues or private inquiries.
              </p>
              <button className="w-full bg-green-600 text-white py-2 px-4 rounded-md hover:bg-green-700 transition-colors">
                Send Email
              </button>
            </div>
            
            <div className="border border-gray-200 rounded-lg p-6">
              <h4 className="font-semibold text-gray-900 mb-3">Office Hours</h4>
              <p className="text-gray-600 text-sm mb-4">
                Join our weekly virtual office hours for live Q&A with the development team.
              </p>
              <button className="w-full bg-purple-600 text-white py-2 px-4 rounded-md hover:bg-purple-700 transition-colors">
                Schedule Session
              </button>
            </div>
          </div>
        </div>
      )
    }
  ];

  const faqs = [
    {
      question: "What is EasyMARL?",
      answer: "EasyMARL is a comprehensive platform for multi-agent reinforcement learning that makes MARL accessible to both beginners and researchers through an intuitive web interface."
    },
    {
      question: "Do I need programming experience?",
      answer: "No! EasyMARL provides a point-and-click interface that requires no command-line or programming experience. However, advanced users can access the full Python API."
    },
    {
      question: "What algorithms are supported?",
      answer: "EasyMARL supports 21+ state-of-the-art MARL algorithms including QMIX, MADDPG, IPPO, COMA, and many more. See our algorithms page for the complete list."
    },
    {
      question: "Can I use my own environments?",
      answer: "Yes! EasyMARL supports custom environment creation. Follow our environment creation guide to build environments tailored to your research needs."
    },
    {
      question: "How do Enhanced Features work?",
      answer: "Enhanced Features provide 10x performance improvements through vectorization, JIT compilation, and advanced optimizations. They're automatically detected and can be enabled through the training interface."
    }
  ];

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="text-center">
        <QuestionMarkCircleIcon className="h-12 w-12 text-blue-600 mx-auto mb-4" />
        <h1 className="text-4xl font-bold text-gray-900 mb-4">
          Help & Documentation
        </h1>
        <p className="text-xl text-gray-600 max-w-3xl mx-auto">
          Everything you need to master multi-agent reinforcement learning with EasyMARL.
        </p>
      </div>

      {/* Tab Navigation */}
      <div className="bg-white rounded-lg shadow-lg overflow-hidden">
        <div className="border-b border-gray-200">
          <nav className="flex space-x-8 px-6" aria-label="Tabs">
            {helpSections.map((section) => {
              const IconComponent = section.icon;
              return (
                <button
                  key={section.id}
                  onClick={() => setActiveTab(section.id)}
                  className={`py-4 px-1 border-b-2 font-medium text-sm flex items-center space-x-2 ${
                    activeTab === section.id
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  <IconComponent className="h-4 w-4" />
                  <span>{section.title}</span>
                </button>
              );
            })}
          </nav>
        </div>

        {/* Tab Content */}
        <div className="p-6">
          {helpSections.find(section => section.id === activeTab)?.content}
        </div>
      </div>

      {/* FAQ Section */}
      <div className="bg-white rounded-lg shadow-lg p-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-6">
          Frequently Asked Questions
        </h2>
        <div className="space-y-6">
          {faqs.map((faq, index) => (
            <div key={index} className="border-b border-gray-200 pb-6 last:border-b-0 last:pb-0">
              <h3 className="text-lg font-semibold text-gray-900 mb-2">
                {faq.question}
              </h3>
              <p className="text-gray-600">
                {faq.answer}
              </p>
            </div>
          ))}
        </div>
      </div>

      {/* Quick Links */}
      <div className="bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg p-8 text-white">
        <h2 className="text-2xl font-bold mb-6">Quick Access</h2>
        <div className="grid md:grid-cols-4 gap-4">
          <a 
            href="/tutorial" 
            className="flex items-center space-x-2 p-3 bg-white bg-opacity-20 rounded-lg hover:bg-opacity-30 transition-colors"
          >
            <BookOpenIcon className="h-5 w-5" />
            <span>Tutorial</span>
          </a>
          <a 
            href="/algorithms" 
            className="flex items-center space-x-2 p-3 bg-white bg-opacity-20 rounded-lg hover:bg-opacity-30 transition-colors"
          >
            <CodeIcon className="h-5 w-5" />
            <span>Algorithms</span>
          </a>
          <a 
            href="/training" 
            className="flex items-center space-x-2 p-3 bg-white bg-opacity-20 rounded-lg hover:bg-opacity-30 transition-colors"
          >
            <AcademicCapIcon className="h-5 w-5" />
            <span>Training</span>
          </a>
          <a 
            href="/research" 
            className="flex items-center space-x-2 p-3 bg-white bg-opacity-20 rounded-lg hover:bg-opacity-30 transition-colors"
          >
            <DocumentTextIcon className="h-5 w-5" />
            <span>Research</span>
          </a>
        </div>
      </div>
    </div>
  );
}

export default HelpPage;
