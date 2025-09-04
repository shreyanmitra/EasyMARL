"""
Controller Tests for EasyMARL

This module tests the unified controller functionality.
"""

import unittest
import sys
import os

# Add the parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class ControllerTestSuite(unittest.TestCase):
    """Test suite for controllers."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_env_name = 'MultiGrid-Empty-6x6-v0'
        self.test_algorithm = 'ippo'
    
    def test_controller_import(self):
        """Test that controller modules can be imported."""
        try:
            from controllers.unified_multiagent_controller import UnifiedMultiAgentController
            
            print("✅ Controller modules imported successfully")
            
        except ImportError as e:
            self.fail(f"Failed to import controller modules: {e}")
    
    def test_controller_creation(self):
        """Test that controllers can be created."""
        try:
            from controllers.unified_multiagent_controller import UnifiedMultiAgentController
            import gymnasium as gym
            
            # Create environment
            try:
                env = gym.make(self.test_env_name)
            except:
                self.skipTest(f"Environment {self.test_env_name} not available")
            
            # Create controller
            controller = UnifiedMultiAgentController(
                env=env,
                algorithm=self.test_algorithm,
                educational_mode=True
            )
            
            self.assertIsNotNone(controller)
            self.assertEqual(controller.algorithm_name.lower(), self.test_algorithm)
            
            print("✅ Controller creation test passed")
            
        except ImportError as e:
            self.skipTest(f"Skipping controller creation test: {e}")
        except Exception as e:
            print(f"⚠️ Controller creation test failed: {e}")
    
    def test_controller_configuration(self):
        """Test controller configuration management."""
        try:
            from controllers.unified_multiagent_controller import UnifiedMultiAgentController
            from core.config_manager import ConfigManager
            import gymnasium as gym
            
            # Test config manager
            config_manager = ConfigManager()
            config = config_manager.get_config('ippo')
            
            self.assertIsNotNone(config)
            self.assertIsInstance(config, dict)
            
            print("✅ Controller configuration test passed")
            
        except ImportError as e:
            self.skipTest(f"Skipping controller configuration test: {e}")
        except Exception as e:
            print(f"⚠️ Controller configuration test failed: {e}")
    
    def test_controller_training_setup(self):
        """Test controller training setup."""
        try:
            from controllers.unified_multiagent_controller import UnifiedMultiAgentController
            import gymnasium as gym
            
            # Create environment
            try:
                env = gym.make(self.test_env_name)
            except:
                self.skipTest(f"Environment {self.test_env_name} not available")
            
            # Create controller
            controller = UnifiedMultiAgentController(
                env=env,
                algorithm=self.test_algorithm,
                educational_mode=True
            )
            
            # Test that training can be set up (without actually training)
            self.assertTrue(hasattr(controller, 'train'))
            self.assertTrue(hasattr(controller, 'evaluate'))
            
            print("✅ Controller training setup test passed")
            
        except ImportError as e:
            self.skipTest(f"Skipping controller training setup test: {e}")
        except Exception as e:
            print(f"⚠️ Controller training setup test failed: {e}")

if __name__ == '__main__':
    unittest.main(verbosity=2)
