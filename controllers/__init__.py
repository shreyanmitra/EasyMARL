"""
EasyMARL Controllers - Unified Multi-Agent Reinforcement Learning

This module provides the unified controller that combines educational clarity,
advanced research features, and high-performance vectorized training.

The UnifiedMultiAgentController replaces the previous separate controllers:
- SimpleMultiAgentController (educational features)
- ModernMultiAgentController (advanced features) 
- VectorizedMultiAgentController (performance features)

All functionality is now available in a single, comprehensive controller.
"""

try:
    from .unified_multiagent_controller import (
        UnifiedMultiAgentController,
        train_unified
    )
    
    # Backward compatibility aliases (deprecated - use UnifiedMultiAgentController)
    SimpleMultiAgentController = UnifiedMultiAgentController
    ModernMultiAgentController = UnifiedMultiAgentController
    VectorizedMultiAgentController = UnifiedMultiAgentController
    
    # Export everything
    __all__ = [
        'UnifiedMultiAgentController',
        'train_unified',
        # Deprecated aliases for backward compatibility
        'SimpleMultiAgentController',
        'ModernMultiAgentController', 
        'VectorizedMultiAgentController'
    ]
    
except ImportError as e:
    print(f"Warning: Could not import controllers: {e}")
    # Handle missing dependencies gracefully
    __all__ = []
