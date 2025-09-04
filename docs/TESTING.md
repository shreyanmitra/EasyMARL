# EasyMARL Testing Documentation

This document provides comprehensive information about testing in the EasyMARL framework.

## Overview

EasyMARL includes a comprehensive test suite to ensure reliability and correctness of all components, from individual algorithms to end-to-end integration.

## Test Structure

### Test Categories

1. **Unit Tests** - Individual component testing
2. **Integration Tests** - System interaction testing
3. **Algorithm Tests** - MARL algorithm correctness
4. **Environment Tests** - Environment validation
5. **Performance Tests** - Benchmark testing

### Test Organization

```
tests/
├── __init__.py           # Test suite runner
├── __main__.py          # CLI test runner
├── test_algorithms.py   # Algorithm functionality tests
├── test_environments.py # Environment tests
├── test_controllers.py  # Controller tests
├── test_utils.py        # Utility function tests
└── test_integration.py  # End-to-end integration tests
```

## Running Tests

### Command Line Interface

```bash
# Run all tests
python -m tests

# Run quick validation tests
python -m tests --quick

# Run specific test suite
python -m tests --suite algorithms

# List available test suites
python -m tests --list

# Verbose output
python -m tests --verbose
```

### Python Interface

```python
from tests import run_all_tests, run_quick_tests

# Run all tests
success = run_all_tests()

# Run quick validation
success = run_quick_tests()
```

### Individual Test Suites

```bash
# Algorithm tests
python -m tests.test_algorithms

# Environment tests  
python -m tests.test_environments

# Controller tests
python -m tests.test_controllers

# Utility tests
python -m tests.test_utils

# Integration tests
python -m tests.test_integration
```

## Test Details

### Algorithm Tests (`test_algorithms.py`)

Tests MARL algorithm functionality:

- **Import Tests**: Verify all algorithms can be imported
- **Creation Tests**: Test algorithm instantiation
- **Training Tests**: Basic training functionality
- **Compatibility Tests**: Algorithm-environment compatibility

**Key Test Cases:**
```python
def test_algorithm_import()           # Algorithm registry loading
def test_algorithm_creation()         # Algorithm class creation
def test_algorithm_basic_training()   # Short training runs
def test_algorithm_compatibility()    # Environment compatibility
```

### Environment Tests (`test_environments.py`)

Tests environment functionality:

- **Import Tests**: Environment module imports
- **Creation Tests**: Environment instantiation
- **Functionality Tests**: Reset, step, observation/action spaces
- **Vectorization Tests**: Parallel environment functionality

**Key Test Cases:**
```python
def test_environment_import()         # Module imports
def test_environment_creation()       # Environment creation
def test_environment_functionality()  # Basic operations
def test_vectorized_environment()     # Parallel environments
```

### Controller Tests (`test_controllers.py`)

Tests unified controller functionality:

- **Import Tests**: Controller module imports
- **Creation Tests**: Controller instantiation
- **Configuration Tests**: Config management
- **Training Setup Tests**: Training pipeline setup

**Key Test Cases:**
```python
def test_controller_import()          # Module imports
def test_controller_creation()        # Controller creation
def test_controller_configuration()   # Config management
def test_controller_training_setup()  # Training setup
```

### Utility Tests (`test_utils.py`)

Tests utility functions and helpers:

- **Import Tests**: Utility module imports
- **Project Root Tests**: Path detection
- **Configuration Tests**: Config management
- **Advanced Features**: JIT compilation, monitoring
- **Research Interface**: Research tools

**Key Test Cases:**
```python
def test_utils_import()               # Module imports
def test_project_root()               # Path detection
def test_config_management()          # Configuration
def test_advanced_features()          # Performance features
def test_research_interface()         # Research tools
```

### Integration Tests (`test_integration.py`)

Tests end-to-end system integration:

- **Full Pipeline Tests**: Complete training workflow
- **CLI Integration**: Command-line interface
- **API Integration**: Flask backend
- **GUI Integration**: Gradio interface

**Key Test Cases:**
```python
def test_full_pipeline()              # End-to-end training
def test_cli_integration()            # CLI functionality
def test_api_integration()            # Backend API
def test_gui_integration()            # GUI interface
```

## Test Configuration

### Quick Tests

Quick tests validate basic functionality without extensive computation:

```python
# Basic import validation
import algorithms
import environments  
import controllers

# Quick controller creation
controller = UnifiedMultiAgentController(
    env=gym.make('MultiGrid-Empty-6x6-v0'),
    algorithm='ippo',
    educational_mode=True
)
```

### Full Test Suite

Full tests include:
- Algorithm training (1 episode)
- Environment functionality validation
- Configuration loading
- Integration testing

### Test Environments

Tests use minimal environments for speed:
- `MultiGrid-Empty-6x6-v0` - Basic testing
- `MultiGrid-Empty-8x8-v0` - Alternative testing

## Continuous Integration

### GitHub Actions

Tests run automatically on:
- Pull requests
- Main branch pushes
- Release tags

### Test Matrix

Tests run across:
- Python 3.11+
- Different OS (Ubuntu, Windows, macOS)
- Different dependency versions

## Writing New Tests

### Test Structure

```python
import unittest
import sys
import os

# Path setup
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class MyTestSuite(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        pass
    
    def test_my_feature(self):
        """Test my feature."""
        # Test implementation
        self.assertTrue(True)
        
    def tearDown(self):
        """Clean up after tests."""
        pass
```

### Best Practices

1. **Fast Tests**: Keep tests quick (< 5 seconds each)
2. **Isolated Tests**: Each test should be independent
3. **Clear Names**: Use descriptive test method names
4. **Error Handling**: Use try/except for optional features
5. **Cleanup**: Properly close environments and resources

### Test Categories

- **Unit Tests**: Test individual functions/classes
- **Integration Tests**: Test component interactions
- **Smoke Tests**: Basic "does it work" validation
- **Performance Tests**: Benchmark critical paths

## Troubleshooting Tests

### Common Issues

**Import Errors:**
```python
try:
    from my_module import MyClass
except ImportError as e:
    self.skipTest(f"Module not available: {e}")
```

**Environment Issues:**
```python
try:
    env = gym.make('MyEnv-v0')
except:
    self.skipTest("Environment not available")
```

**Optional Dependencies:**
```python
try:
    import torch
except ImportError:
    self.skipTest("PyTorch not available")
```

### Debug Mode

Run tests with debug information:

```bash
python -m tests --verbose
python -m unittest discover -v tests/
```

## Test Coverage

### Coverage Reports

Generate coverage reports:

```bash
pip install coverage
coverage run -m tests
coverage report
coverage html  # Generate HTML report
```

### Target Coverage

- **Core Modules**: 80%+ coverage
- **Algorithms**: 70%+ coverage  
- **Utilities**: 90%+ coverage
- **Overall**: 75%+ coverage

## Performance Testing

### Benchmark Tests

Performance benchmarks for:
- Algorithm training speed
- Environment step time
- Memory usage
- GPU utilization

### Profiling

```python
import cProfile
import pstats

# Profile test function
cProfile.run('test_function()', 'profile_output.prof')
stats = pstats.Stats('profile_output.prof')
stats.sort_stats('cumulative').print_stats(10)
```

For more information, see the main documentation at: https://shreyanmitra.github.io/EasyMARL
