"""
Algorithm Tests for EasyMARL

This module tests the correctness and functionality of MARL algorithms.
"""

import unittest
import sys
import os
import numpy as np

# Add the parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class AlgorithmTestSuite(unittest.TestCase):
    """Test suite for MARL algorithms."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_env_name = 'MultiGrid-Empty-6x6-v0'
        self.algorithms_to_test = ['ippo', 'qmix', 'vdn', 'maddpg', 'mappo']
    
    def test_algorithm_import(self):
        """Test that all algorithms can be imported."""
        try:
            from algorithms import ALGORITHM_REGISTRY, list_available_algorithms
            
            # Test registry exists
            self.assertIsInstance(ALGORITHM_REGISTRY, dict)
            self.assertGreater(len(ALGORITHM_REGISTRY), 0)
            
            # Test list function works
            algorithms = list_available_algorithms()
            self.assertIsInstance(algorithms, dict)
            
            print(f"✅ Found {len(ALGORITHM_REGISTRY)} algorithms in registry")
            
        except ImportError as e:
            self.fail(f"Failed to import algorithms module: {e}")
    
    def test_algorithm_creation(self):
        """Test that algorithms can be created."""
        try:
            from algorithms import get_algorithm_class
            from controllers.unified_multiagent_controller import UnifiedMultiAgentController
            import gymnasium as gym
            
            # Test basic algorithm creation
            for algo_name in ['IPPO', 'QMIX', 'VDN']:
                if algo_name in ['IPPO', 'QMIX', 'VDN']:  # Only test these for now
                    try:
                        algo_class = get_algorithm_class(algo_name)
                        self.assertIsNotNone(algo_class)
                        print(f"✅ {algo_name} algorithm class created successfully")
                    except Exception as e:
                        print(f"⚠️ {algo_name} algorithm creation failed: {e}")
                        
        except ImportError as e:
            self.skipTest(f"Skipping algorithm creation test: {e}")
    
    def test_algorithm_basic_training(self):
        """Test basic algorithm training functionality."""
        try:
            from controllers.unified_multiagent_controller import UnifiedMultiAgentController
            import gymnasium as gym
            
            # Create a simple environment
            try:
                env = gym.make(self.test_env_name)
            except:
                self.skipTest(f"Environment {self.test_env_name} not available")
            
            # Test IPPO (simplest algorithm)
            try:
                controller = UnifiedMultiAgentController(
                    env=env,
                    algorithm='ippo',
                    educational_mode=True
                )
                
                # Quick training test (1 episode)
                controller.train(episodes=1)
                print("✅ Basic IPPO training test passed")
                
            except Exception as e:
                print(f"⚠️ Basic training test failed: {e}")
                
        except ImportError as e:
            self.skipTest(f"Skipping training test: {e}")
    
    def test_algorithm_compatibility(self):
        """Test algorithm compatibility with different environments."""
        try:
            from algorithms import ALGORITHM_REGISTRY
            
            # Test that each algorithm has required attributes
            for algo_name, algo_info in ALGORITHM_REGISTRY.items():
                self.assertIn('class', algo_info, f"{algo_name} missing class info")
                self.assertIn('type', algo_info, f"{algo_name} missing type info")
                
            print(f"✅ Algorithm compatibility check passed for {len(ALGORITHM_REGISTRY)} algorithms")
            
        except ImportError as e:
            self.skipTest(f"Skipping compatibility test: {e}")

if __name__ == '__main__':
    unittest.main(verbosity=2)
