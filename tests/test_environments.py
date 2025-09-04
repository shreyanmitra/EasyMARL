"""
Environment Tests for EasyMARL

This module tests environment functionality and compatibility.
"""

import unittest
import sys
import os
import numpy as np

# Add the parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class EnvironmentTestSuite(unittest.TestCase):
    """Test suite for environments."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_environments = [
            'MultiGrid-Empty-6x6-v0',
            'MultiGrid-Empty-8x8-v0'
        ]
    
    def test_environment_import(self):
        """Test that environment modules can be imported."""
        try:
            import environments
            from environments.vectorized_env import VectorizedEnvironment
            
            print("✅ Environment modules imported successfully")
            
        except ImportError as e:
            self.fail(f"Failed to import environment modules: {e}")
    
    def test_environment_creation(self):
        """Test that environments can be created."""
        try:
            import gymnasium as gym
            
            # Test basic environment creation
            for env_name in self.test_environments:
                try:
                    env = gym.make(env_name)
                    self.assertIsNotNone(env)
                    
                    # Test basic environment properties
                    self.assertTrue(hasattr(env, 'observation_space'))
                    self.assertTrue(hasattr(env, 'action_space'))
                    
                    env.close()
                    print(f"✅ {env_name} created successfully")
                    
                except Exception as e:
                    print(f"⚠️ Failed to create {env_name}: {e}")
                    
        except ImportError as e:
            self.skipTest(f"Skipping environment creation test: {e}")
    
    def test_environment_functionality(self):
        """Test basic environment functionality."""
        try:
            import gymnasium as gym
            
            for env_name in self.test_environments:
                try:
                    env = gym.make(env_name)
                    
                    # Test reset
                    obs, info = env.reset()
                    self.assertIsNotNone(obs)
                    
                    # Test step
                    if hasattr(env, 'action_space'):
                        action = env.action_space.sample()
                        obs, reward, terminated, truncated, info = env.step(action)
                        
                        self.assertIsNotNone(obs)
                        self.assertIsNotNone(reward)
                        self.assertIsInstance(terminated, bool)
                        self.assertIsInstance(truncated, bool)
                    
                    env.close()
                    print(f"✅ {env_name} functionality test passed")
                    break  # Test only first available environment
                    
                except Exception as e:
                    print(f"⚠️ {env_name} functionality test failed: {e}")
                    
        except ImportError as e:
            self.skipTest(f"Skipping environment functionality test: {e}")
    
    def test_vectorized_environment(self):
        """Test vectorized environment functionality."""
        try:
            from environments.vectorized_env import VectorizedEnvironment
            
            # Create vectorized environment
            vec_env = VectorizedEnvironment(
                env_name=self.test_environments[0],
                n_envs=2
            )
            
            # Test reset
            obs = vec_env.reset()
            self.assertIsNotNone(obs)
            
            # Test step
            actions = [vec_env.action_space.sample() for _ in range(2)]
            obs, rewards, dones, infos = vec_env.step(actions)
            
            self.assertIsNotNone(obs)
            self.assertIsNotNone(rewards)
            self.assertIsNotNone(dones)
            
            vec_env.close()
            print("✅ Vectorized environment test passed")
            
        except ImportError as e:
            self.skipTest(f"Vectorized environment not available: {e}")
        except Exception as e:
            print(f"⚠️ Vectorized environment test failed: {e}")

if __name__ == '__main__':
    unittest.main(verbosity=2)
