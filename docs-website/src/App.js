import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import HomePage from './pages/HomePage';
import DocsPage from './pages/DocsPage';
import AlgorithmsPage from './pages/AlgorithmsPage';
import TutorialsPage from './pages/TutorialsPage';
import ExamplesPage from './pages/ExamplesPage';
import APIReferencePage from './pages/APIReferencePage';
import ComparisonPage from './pages/ComparisonPage';
import Footer from './components/Footer';
import ScrollIndicator from './components/ScrollIndicator';
import './index.css';

function App() {
  return (
    <Router basename="/EasyMARL">
      <div className="App">
        <ScrollIndicator />
        <Navbar />
        <main className="min-h-screen">
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/docs/*" element={<DocsPage />} />
            <Route path="/algorithms" element={<AlgorithmsPage />} />
            <Route path="/tutorials" element={<TutorialsPage />} />
            <Route path="/examples" element={<ExamplesPage />} />
            <Route path="/api" element={<APIReferencePage />} />
            <Route path="/comparison" element={<ComparisonPage />} />
          </Routes>
        </main>
        <Footer />
      </div>
    </Router>
  );
}

export default App;
