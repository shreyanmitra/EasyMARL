import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import Navbar from './components/Navbar';
import HomePage from './pages/HomePage';
import TrainingPage from './pages/TrainingPage';
import AlgorithmsPage from './pages/AlgorithmsPage';
import EnvironmentsPage from './pages/EnvironmentsPage';
import HelpPage from './pages/HelpPage';
import './App.css';

/**
 * Main EasyMARL React Application
 * 
 * This is the root component that sets up routing and global layout
 * for the EasyMARL web interface. It provides the same functionality
 * as the Python GUI but with modern React architecture.
 */
function App() {
  return (
    <Router basename="/EasyMARL">
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
        {/* Navigation Bar */}
        <Navbar />
        
        {/* Main Content Area */}
        <main className="container mx-auto px-4 py-8">
          <Routes>
            {/* Home Page - Welcome and Overview */}
            <Route path="/" element={<HomePage />} />
            
            {/* Training Page - Start and Monitor Training */}
            <Route path="/training" element={<TrainingPage />} />
            
            {/* Algorithms Page - Browse and Learn About Algorithms */}
            <Route path="/algorithms" element={<AlgorithmsPage />} />
            
            {/* Environments Page - Explore Available Environments */}
            <Route path="/environments" element={<EnvironmentsPage />} />
            
            {/* Help Page - Documentation and Tutorials */}
            <Route path="/help" element={<HelpPage />} />
          </Routes>
        </main>
        
        {/* Toast Notifications for User Feedback */}
        <Toaster 
          position="top-right"
          toastOptions={{
            duration: 4000,
            style: {
              background: '#363636',
              color: '#fff',
            },
          }}
        />
      </div>
    </Router>
  );
}

export default App;
