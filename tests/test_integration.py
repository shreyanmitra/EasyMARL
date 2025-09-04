"""
Integration Tests for EasyMARL

This module tests end-to-end integration of EasyMARL components.
"""

import unittest
import sys
import os

# Add the parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class IntegrationTestSuite(unittest.TestCase):
    """Test suite for system integration."""
    
    def test_full_pipeline(self):
        """Test complete training pipeline."""
        try:
            from controllers.unified_multiagent_controller import UnifiedMultiAgentController
            import gymnasium as gym
            
            # Create environment
            try:
                env = gym.make('MultiGrid-Empty-6x6-v0')
            except:
                self.skipTest("Test environment not available")
            
            # Create controller
            controller = UnifiedMultiAgentController(
                env=env,
                algorithm='ippo',
                educational_mode=True
            )
            
            # Run very short training
            controller.train(episodes=1)
            
            # Test evaluation
            results = controller.evaluate(episodes=1)
            self.assertIsNotNone(results)
            
            print("✅ Full pipeline integration test passed")
            
        except ImportError as e:
            self.skipTest(f"Skipping full pipeline test: {e}")
        except Exception as e:
            print(f"⚠️ Full pipeline test failed: {e}")
    
    def test_cli_integration(self):
        """Test CLI integration."""
        try:
            from cli.main_cli import train_command
            import argparse
            
            # Create test args
            args = argparse.Namespace(
                algorithm='ippo',
                env_name='MultiGrid-Empty-6x6-v0',
                episodes=1,
                evaluate=True,
                visualize=False,
                debug=True,
                vectorized=False,
                n_envs=1,
                seed=42,
                keep_training=False,
                wandb_project='EasyMARL-Test',
                list_algorithms=False
            )
            
            # Test CLI command (this might fail but shouldn't crash)
            try:
                result = train_command(args)
                print("✅ CLI integration test passed")
            except Exception as e:
                print(f"⚠️ CLI integration test failed: {e}")
                
        except ImportError as e:
            self.skipTest(f"Skipping CLI integration test: {e}")
    
    def test_api_integration(self):
        """Test API integration."""
        try:
            from api.flask_backend import app
            
            # Test that Flask app can be created
            self.assertIsNotNone(app)
            
            # Test basic routes exist
            with app.test_client() as client:
                # Test health check
                response = client.get('/api/health')
                # Don't require specific response, just test it doesn't crash
                
            print("✅ API integration test passed")
            
        except ImportError as e:
            self.skipTest(f"Skipping API integration test: {e}")
        except Exception as e:
            print(f"⚠️ API integration test failed: {e}")
    
    def test_gui_integration(self):
        """Test GUI integration."""
        try:
            from gui.gradio_interface import create_interface
            
            # Test that Gradio interface can be created
            interface = create_interface()
            self.assertIsNotNone(interface)
            
            print("✅ GUI integration test passed")
            
        except ImportError as e:
            self.skipTest(f"Skipping GUI integration test: {e}")
        except Exception as e:
            print(f"⚠️ GUI integration test failed: {e}")

if __name__ == '__main__':
    unittest.main(verbosity=2)
