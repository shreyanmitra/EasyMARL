#!/usr/bin/env python3
"""
EasyMARL Test Runner

Command-line interface for running EasyMARL tests.

Usage:
    python -m tests                    # Run all tests
    python -m tests --quick            # Run quick validation tests
    python -m tests --suite algorithms # Run specific test suite
    python -m tests --list             # List available test suites
"""

import sys
import os
import argparse

# Add the parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_test_parser():
    """Create argument parser for test runner."""
    parser = argparse.ArgumentParser(
        description="EasyMARL Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m tests                     # Run all tests
  python -m tests --quick             # Quick validation
  python -m tests --suite algorithms  # Specific suite
  python -m tests --verbose           # Detailed output
        """
    )
    
    parser.add_argument('--suite', type=str, default=None,
                       choices=['algorithms', 'environments', 'controllers', 'utils', 'integration'],
                       help='Run specific test suite')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick validation tests only')
    parser.add_argument('--list', action='store_true',
                       help='List available test suites')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose test output')
    
    return parser

def list_test_suites():
    """List available test suites."""
    print("🧪 Available EasyMARL Test Suites:")
    print("=" * 40)
    print("  • algorithms    - MARL algorithm tests")
    print("  • environments  - Environment functionality tests")
    print("  • controllers   - Controller and training tests")
    print("  • utils         - Utility function tests")
    print("  • integration   - End-to-end integration tests")
    print("")
    print("Usage:")
    print("  python -m tests --suite algorithms")
    print("  python -m tests --quick")
    print("  python -m tests  # Run all tests")

def run_specific_suite(suite_name, verbose=False):
    """Run a specific test suite."""
    import unittest
    
    suite_map = {
        'algorithms': 'test_algorithms.AlgorithmTestSuite',
        'environments': 'test_environments.EnvironmentTestSuite', 
        'controllers': 'test_controllers.ControllerTestSuite',
        'utils': 'test_utils.UtilsTestSuite',
        'integration': 'test_integration.IntegrationTestSuite'
    }
    
    if suite_name not in suite_map:
        print(f"❌ Unknown test suite: {suite_name}")
        return False
    
    print(f"🧪 Running {suite_name} test suite...")
    
    try:
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromName(suite_map[suite_name])
        
        verbosity = 2 if verbose else 1
        runner = unittest.TextTestRunner(verbosity=verbosity)
        result = runner.run(suite)
        
        return result.wasSuccessful()
        
    except ImportError as e:
        print(f"❌ Failed to import test suite: {e}")
        return False
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        return False

def run_quick_tests():
    """Run quick validation tests."""
    from tests import run_quick_tests
    return run_quick_tests()

def run_all_tests(verbose=False):
    """Run all available test suites."""
    print("🧪 Running All EasyMARL Tests...")
    print("=" * 40)
    
    suites = ['algorithms', 'environments', 'controllers', 'utils', 'integration']
    results = {}
    
    for suite in suites:
        print(f"\n📋 Testing {suite}...")
        results[suite] = run_specific_suite(suite, verbose)
    
    # Summary
    print("\n📊 Test Results Summary:")
    print("=" * 40)
    
    total_passed = 0
    for suite, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {suite:12} {status}")
        if passed:
            total_passed += 1
    
    success_rate = (total_passed / len(suites)) * 100
    print(f"\nOverall: {total_passed}/{len(suites)} suites passed ({success_rate:.1f}%)")
    
    return total_passed == len(suites)

def main():
    """Main test runner entry point."""
    parser = create_test_parser()
    args = parser.parse_args()
    
    if args.list:
        list_test_suites()
        return 0
    elif args.quick:
        success = run_quick_tests()
    elif args.suite:
        success = run_specific_suite(args.suite, args.verbose)
    else:
        success = run_all_tests(args.verbose)
    
    return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main())
