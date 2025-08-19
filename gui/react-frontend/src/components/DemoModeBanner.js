import React from 'react';
import { ExclamationTriangleIcon, EyeIcon } from '@heroicons/react/24/outline';

/**
 * Demo Mode Banner Component
 * 
 * Displays a prominent banner when the app is running in demo/read-only mode
 * to inform users that training operations are disabled.
 */
const DemoModeBanner = ({ demoMode = false, readOnly = false }) => {
  // Don't show banner if not in demo mode
  if (!demoMode && !readOnly) {
    return null;
  }

  return (
    <div className="bg-yellow-50 border-l-4 border-yellow-400 p-4 mb-6">
      <div className="flex items-center">
        <div className="flex-shrink-0">
          {readOnly ? (
            <EyeIcon className="h-5 w-5 text-yellow-400" aria-hidden="true" />
          ) : (
            <ExclamationTriangleIcon className="h-5 w-5 text-yellow-400" aria-hidden="true" />
          )}
        </div>
        <div className="ml-3">
          <p className="text-sm text-yellow-700">
            <span className="font-medium">
              {readOnly ? 'Read-Only Demo Mode' : 'Demo Mode'}
            </span>
            {' - '}
            This is a demonstration environment. Training operations are disabled.
            {' '}
            <a 
              href="https://github.com/YourUsername/EasyMARL" 
              className="font-medium underline text-yellow-700 hover:text-yellow-600"
              target="_blank"
              rel="noopener noreferrer"
            >
              Use the full repository for development →
            </a>
          </p>
        </div>
      </div>
    </div>
  );
};

export default DemoModeBanner;
