import React, { useState, useEffect } from 'react';
import { Copy, Check } from 'lucide-react';

// Import Prism.js for syntax highlighting
import Prism from 'prismjs';
import 'prismjs/themes/prism-tomorrow.css'; // Dark theme
import 'prismjs/components/prism-python';
import 'prismjs/components/prism-javascript';
import 'prismjs/components/prism-yaml';
import 'prismjs/components/prism-json';
import 'prismjs/components/prism-bash';

const CodeBlock = ({ code, language = 'python', title, className }) => {
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    // Highlight syntax after component mounts
    Prism.highlightAll();
  }, [code, language]);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy code:', err);
    }
  };

  // Map common language aliases
  const languageMap = {
    'python': 'python',
    'py': 'python', 
    'javascript': 'javascript',
    'js': 'javascript',
    'yaml': 'yaml',
    'yml': 'yaml',
    'json': 'json',
    'bash': 'bash',
    'shell': 'bash',
    'terminal': 'bash'
  };

  const prismLanguage = languageMap[language] || 'python';

  return (
    <div className={`relative group ${className}`}>
      {title && (
        <div className="bg-gray-800 text-white px-4 py-2 text-sm font-medium rounded-t-lg border-b border-gray-700 flex items-center justify-between">
          <span>{title}</span>
          <span className="text-xs bg-gray-700 px-2 py-1 rounded text-gray-300">
            {language.toUpperCase()}
          </span>
        </div>
      )}
      
      <div className="relative">
        <button
          onClick={handleCopy}
          className="absolute top-4 right-4 p-2 bg-gray-700 hover:bg-gray-600 text-white rounded-lg opacity-0 group-hover:opacity-100 transition-all duration-200 z-10 flex items-center space-x-1"
          title={copied ? 'Copied!' : 'Copy code'}
        >
          {copied ? (
            <>
              <Check className="w-4 h-4" />
              <span className="text-xs">Copied!</span>
            </>
          ) : (
            <>
              <Copy className="w-4 h-4" />
              <span className="text-xs">Copy</span>
            </>
          )}
        </button>
        
        <pre className={`line-numbers bg-gray-900 text-gray-300 p-4 overflow-x-auto font-mono text-sm leading-relaxed ${title ? 'rounded-b-lg' : 'rounded-lg'} border border-gray-700`}>
          <code className={`language-${prismLanguage}`}>
            {code}
          </code>
        </pre>
      </div>
    </div>
  );
};

export default CodeBlock;
