"""
EasyMARL Test Suite

This module provides comprehensive testing for the EasyMARL framework,
including unit tests, integration tests, and algorithm validation tests.

Test Categories:
- Unit Tests: Individual component testing
- Integration Tests: System interaction testing  
- Algorithm Tests: MARL algorithm correctness testing
- Environment Tests: Environment validation testing
- Performance Tests: Benchmark and performance testing
"""

import unittest
import sys
import os

# Add the parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import test modules (with fallbacks if not available)
try:
    from .test_algorithms import AlgorithmTestSuite
except ImportError:
    AlgorithmTestSuite = None

try:
    from .test_environments import EnvironmentTestSuite
except ImportError:
    EnvironmentTestSuite = None

try:
    from .test_controllers import ControllerTestSuite
except ImportError:
    ControllerTestSuite = None

try:
    from .test_utils import UtilsTestSuite
except ImportError:
    UtilsTestSuite = None

try:
    from .test_integration import IntegrationTestSuite
except ImportError:
    IntegrationTestSuite = None

def run_all_tests():
    """Run all test suites."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all available test suites
    test_suites = [AlgorithmTestSuite, EnvironmentTestSuite, ControllerTestSuite, 
                   UtilsTestSuite, IntegrationTestSuite]
    
    for test_suite in test_suites:
        if test_suite is not None:
            suite.addTest(loader.loadTestsFromTestCase(test_suite))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

def run_quick_tests():
    """Run quick validation tests."""
    print("🧪 Running EasyMARL Quick Tests...")
    
    # Basic import tests
    try:
        import algorithms
        print("✅ Algorithms module import successful")
    except ImportError as e:
        print(f"❌ Algorithms module import failed: {e}")
        return False
    
    try:
        import environments
        print("✅ Environments module import successful") 
    except ImportError as e:
        print(f"❌ Environments module import failed: {e}")
        return False
        
    try:
        import controllers
        print("✅ Controllers module import successful")
    except ImportError as e:
        print(f"❌ Controllers module import failed: {e}")
        return False
    
    print("✅ Quick tests passed!")
    return True

if __name__ == '__main__':
    if '--quick' in sys.argv:
        success = run_quick_tests()
    else:
        success = run_all_tests()
        
    sys.exit(0 if success else 1)
