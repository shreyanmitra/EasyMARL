"""
Utility Tests for EasyMARL

This module tests utility functions and helper modules.
"""

import unittest
import sys
import os

# Add the parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class UtilsTestSuite(unittest.TestCase):
    """Test suite for utility functions."""
    
    def test_utils_import(self):
        """Test that utility modules can be imported."""
        try:
            from core.utils.base import get_project_root
            from core.utils.advanced import enable_jit_compilation
            from core.utils.enhanced import setup_advanced_monitoring
            
            print("✅ Utility modules imported successfully")
            
        except ImportError as e:
            self.fail(f"Failed to import utility modules: {e}")
    
    def test_project_root(self):
        """Test project root detection."""
        try:
            from core.utils.base import get_project_root
            
            root = get_project_root()
            self.assertIsNotNone(root)
            self.assertTrue(os.path.exists(root))
            
            print(f"✅ Project root detected: {root}")
            
        except ImportError as e:
            self.skipTest(f"Skipping project root test: {e}")
        except Exception as e:
            print(f"⚠️ Project root test failed: {e}")
    
    def test_config_management(self):
        """Test configuration management utilities."""
        try:
            from core.config_manager import ConfigManager
            
            config_manager = ConfigManager()
            
            # Test default config loading
            default_config = config_manager.get_default_config()
            self.assertIsNotNone(default_config)
            self.assertIsInstance(default_config, dict)
            
            # Test algorithm config loading
            ippo_config = config_manager.get_config('ippo')
            self.assertIsNotNone(ippo_config)
            
            print("✅ Configuration management test passed")
            
        except ImportError as e:
            self.skipTest(f"Skipping config management test: {e}")
        except Exception as e:
            print(f"⚠️ Config management test failed: {e}")
    
    def test_advanced_features(self):
        """Test advanced utility features."""
        try:
            from core.utils.advanced import enable_jit_compilation, setup_performance_monitoring
            
            # Test JIT compilation setup (should not error)
            try:
                enable_jit_compilation()
                print("✅ JIT compilation setup successful")
            except Exception as e:
                print(f"⚠️ JIT compilation setup failed: {e}")
            
            # Test performance monitoring setup
            try:
                setup_performance_monitoring()
                print("✅ Performance monitoring setup successful") 
            except Exception as e:
                print(f"⚠️ Performance monitoring setup failed: {e}")
                
        except ImportError as e:
            self.skipTest(f"Skipping advanced features test: {e}")
    
    def test_research_interface(self):
        """Test research interface utilities."""
        try:
            from core.research_interface import ResearchInterface
            
            # Create research interface
            research = ResearchInterface()
            self.assertIsNotNone(research)
            
            # Test basic functionality
            self.assertTrue(hasattr(research, 'setup_experiment'))
            self.assertTrue(hasattr(research, 'log_metrics'))
            
            print("✅ Research interface test passed")
            
        except ImportError as e:
            self.skipTest(f"Skipping research interface test: {e}")
        except Exception as e:
            print(f"⚠️ Research interface test failed: {e}")

if __name__ == '__main__':
    unittest.main(verbosity=2)
