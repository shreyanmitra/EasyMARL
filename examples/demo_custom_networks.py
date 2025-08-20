"""
EasyMARL Custom Networks Example

This script demonstrates how to use different custom neural network architectures
with EasyMARL algorithms. It shows practical examples of:

1. Built-in custom networks (feedforward, convolutional, attention, residual)
2. Configuring networks for different scenarios 
3. Comparing network performance
4. Creating your own custom networks

Run this script to see custom networks in action!
"""

import torch
import numpy as np
from typing import Dict, List
import time

# EasyMARL imports
from core.config_manager import ConfigManager
from environments.vectorized_env import VectorizedMultiGridEnv

# Try to import algorithm - fallback if not available
try:
    from algorithms.model_free.policy_based.discrete_action.ippo import IPPO
    ALGORITHM_AVAILABLE = True
except ImportError:
    print("Algorithm not available for demonstration")
    ALGORITHM_AVAILABLE = False

# Custom network imports
try:
    from networks.base_network import get_network, NetworkFactory
    from networks.custom_networks import (
        FeedForwardNetwork, ConvolutionalNetwork, 
        AttentionNetwork, ResidualNetwork
    )
    NETWORKS_AVAILABLE = True
except ImportError:
    print("Custom networks not available")
    NETWORKS_AVAILABLE = False


def demo_network_creation():
    """
    Demonstrate how to create different types of custom networks.
    """
    print("🔧 Custom Network Creation Demo")
    print("=" * 50)
    
    if not NETWORKS_AVAILABLE:
        print("Custom networks not available. Please check installation.")
        return
    
    # Define a sample observation space (typical for MultiGrid)
    obs_space = {
        'image': (7, 7, 3),    # 7x7 grid with 3 channels
        'direction': 4         # 4 possible directions
    }
    
    # Basic configuration
    config = {
        'hidden_dim': 128,
        'learning_rate': 0.0003
    }
    
    action_space = 6  # Typical MultiGrid actions
    
    # 1. Feedforward Network
    print("\n1️⃣ Creating FeedForward Network...")
    try:
        ff_config = {**config, 'num_layers': 3, 'activation': 'relu'}
        ff_network = NetworkFactory.create_network(
            'feedforward', obs_space, ff_config, action_space
        )
        print(f"   ✅ Created with {sum(p.numel() for p in ff_network.parameters()):,} parameters")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # 2. Convolutional Network
    print("\n2️⃣ Creating Convolutional Network...")
    try:
        conv_config = {**config, 'conv_layers': [32, 64, 64], 'use_pooling': True}
        conv_network = NetworkFactory.create_network(
            'convolutional', obs_space, conv_config, action_space
        )
        print(f"   ✅ Created with {sum(p.numel() for p in conv_network.parameters()):,} parameters")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # 3. Attention Network
    print("\n3️⃣ Creating Attention Network...")
    try:
        att_config = {**config, 'num_heads': 8, 'attention_dim': 128}
        att_network = NetworkFactory.create_network(
            'attention', obs_space, att_config, action_space
        )
        print(f"   ✅ Created with {sum(p.numel() for p in att_network.parameters()):,} parameters")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # 4. Residual Network
    print("\n4️⃣ Creating Residual Network...")
    try:
        res_config = {**config, 'num_blocks': 3, 'use_batch_norm': True}
        res_network = NetworkFactory.create_network(
            'residual', obs_space, res_config, action_space
        )
        print(f"   ✅ Created with {sum(p.numel() for p in res_network.parameters()):,} parameters")
    except Exception as e:
        print(f"   ❌ Error: {e}")


def demo_network_forward_pass():
    """
    Demonstrate forward passes through different networks.
    """
    print("\n\n🚀 Network Forward Pass Demo")
    print("=" * 50)
    
    if not NETWORKS_AVAILABLE:
        print("Custom networks not available.")
        return
    
    # Create sample observation
    sample_obs = {
        'image': torch.randn(1, 7, 7, 3),  # Batch size 1
        'direction': torch.tensor([2])
    }
    
    obs_space = {'image': (7, 7, 3), 'direction': 4}
    config = {'hidden_dim': 128}
    action_space = 6
    
    networks = ['feedforward', 'convolutional', 'attention', 'residual']
    
    for net_type in networks:
        print(f"\n🧠 Testing {net_type.title()} Network:")
        try:
            # Create network
            network = NetworkFactory.create_network(
                net_type, obs_space, config, action_space
            )
            
            # Forward pass
            start_time = time.time()
            output = network(sample_obs)
            forward_time = time.time() - start_time
            
            print(f"   Output shape: {output.shape}")
            print(f"   Forward time: {forward_time*1000:.2f}ms")
            print(f"   Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")


def demo_ippo_with_custom_networks():
    """
    Demonstrate IPPO training with different custom networks.
    """
    print("\n\n🎯 IPPO + Custom Networks Demo")
    print("=" * 50)
    
    if not ALGORITHM_AVAILABLE or not NETWORKS_AVAILABLE:
        print("Required components not available.")
        return
    
    # Create a simple environment for testing
    try:
        env = VectorizedMultiGridEnv('MultiGrid-Empty-6x6-v0', n_envs=1)
    except Exception as e:
        print(f"Could not create environment: {e}")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test different network configurations
    network_configs = {
        'FeedForward': {
            'network_type': 'feedforward',
            'hidden_dim': 128,
            'num_layers': 3,
            'rollout_length': 32,  # Short for demo
            'learning_rate': 0.003,
            'gamma': 0.99
        },
        'Convolutional': {
            'network_type': 'convolutional',
            'hidden_dim': 128,
            'conv_layers': [32, 64],
            'rollout_length': 32,
            'learning_rate': 0.003,
            'gamma': 0.99
        },
        'Attention': {
            'network_type': 'attention',
            'hidden_dim': 128,
            'num_heads': 4,
            'attention_dim': 64,
            'rollout_length': 32,
            'learning_rate': 0.001,  # Lower LR for attention
            'gamma': 0.99
        }
    }
    
    for net_name, config in network_configs.items():
        print(f"\n🤖 Testing IPPO with {net_name} Network:")
        try:
            # Create IPPO algorithm
            algorithm = IPPO(env, config, device)
            
            # Run a few training steps
            print(f"   Running 3 training steps...")
            for step in range(3):
                # Collect rollout
                rollout = algorithm.collect_rollout(env)
                
                # Train on rollout
                metrics = {}
                for agent in algorithm.agents:
                    agent_metrics = agent.update()
                    for key, value in agent_metrics.items():
                        if key not in metrics:
                            metrics[key] = []
                        metrics[key].append(value)
                
                # Print progress
                avg_reward = np.mean(rollout.get('rewards', [0]))
                print(f"   Step {step+1}: avg_reward={avg_reward:.3f}")
            
            print(f"   ✅ {net_name} network training successful!")
            
        except Exception as e:
            print(f"   ❌ Error with {net_name} network: {e}")


def demo_performance_comparison():
    """
    Compare computational performance of different networks.
    """
    print("\n\n⚡ Network Performance Comparison")
    print("=" * 50)
    
    if not NETWORKS_AVAILABLE:
        print("Custom networks not available.")
        return
    
    # Setup
    obs_space = {'image': (7, 7, 3), 'direction': 4}
    config = {'hidden_dim': 128}
    action_space = 6
    n_forward_passes = 100
    
    # Create sample batch
    batch_obs = {
        'image': torch.randn(32, 7, 7, 3),  # Batch of 32
        'direction': torch.randint(0, 4, (32,))
    }
    
    networks = ['feedforward', 'convolutional', 'attention', 'residual']
    results = {}
    
    for net_type in networks:
        print(f"\n📊 Benchmarking {net_type.title()} Network:")
        try:
            # Create network
            network = NetworkFactory.create_network(
                net_type, obs_space, config, action_space
            )
            network.eval()  # Set to evaluation mode
            
            # Warm up
            with torch.no_grad():
                for _ in range(10):
                    _ = network(batch_obs)
            
            # Benchmark
            start_time = time.time()
            with torch.no_grad():
                for _ in range(n_forward_passes):
                    output = network(batch_obs)
            
            total_time = time.time() - start_time
            avg_time = total_time / n_forward_passes
            throughput = (32 * n_forward_passes) / total_time  # samples per second
            
            results[net_type] = {
                'avg_time': avg_time,
                'throughput': throughput,
                'parameters': sum(p.numel() for p in network.parameters())
            }
            
            print(f"   Average time: {avg_time*1000:.2f}ms")
            print(f"   Throughput: {throughput:.0f} samples/sec")
            print(f"   Parameters: {results[net_type]['parameters']:,}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Summary
    if results:
        print(f"\n📈 Performance Summary:")
        print(f"{'Network':<15} {'Time (ms)':<12} {'Throughput':<12} {'Parameters':<12}")
        print("-" * 55)
        for net_type, metrics in results.items():
            print(f"{net_type.title():<15} {metrics['avg_time']*1000:<12.2f} "
                  f"{metrics['throughput']:<12.0f} {metrics['parameters']:<12,}")


def demo_custom_user_network():
    """
    Demonstrate how users can create their own custom networks.
    """
    print("\n\n🛠️ Custom User Network Demo")
    print("=" * 50)
    
    if not NETWORKS_AVAILABLE:
        print("Custom networks not available.")
        return
    
    # Example: Create a simple custom network class
    print("Creating a simple custom network class...")
    
    custom_network_code = '''
from networks.base_network import MultiGridCompatibleNetwork
import torch.nn as nn

class MyCustomNetwork(MultiGridCompatibleNetwork):
    """
    Example custom network that users can create.
    """
    
    def __init__(self, obs_space, config, action_space, n_agents=1, agent_id=0):
        super().__init__(obs_space, config, action_space, n_agents, agent_id)
    
    def _build_network(self):
        """Build your custom architecture here."""
        # Get input features
        feature_dim = self._calculate_feature_dim()
        
        # Create custom architecture
        self.network = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_space)
        )
    
    def _calculate_feature_dim(self):
        # Calculate total input features
        return (self.image_height * self.image_width * self.image_channels + 
                (1 if self.has_direction else 0))
    
    def forward(self, observations):
        features = self._extract_features(self.process_observations(observations))
        return self.network(features)
'''
    
    print("Custom network code structure:")
    print(custom_network_code)
    
    print("\nTo use your custom network:")
    print("1. Save the network class in a Python file (e.g., 'my_networks.py')")
    print("2. Set network_type to 'my_networks.MyCustomNetwork' in your config")
    print("3. EasyMARL will automatically import and use your network!")
    
    config_example = '''
config = {
    'network_type': 'my_networks.MyCustomNetwork',
    'hidden_dim': 256,
    'learning_rate': 0.0003,
    'custom_param': 'your_value'  # Your custom parameters
}
'''
    print(f"\nExample configuration:\n{config_example}")


def main():
    """
    Run all custom network demonstrations.
    """
    print("🌟 Welcome to EasyMARL Custom Networks Demo!")
    print("This demo shows how to use custom neural networks with EasyMARL.")
    print("\n" + "=" * 60)
    
    # Run all demonstrations
    try:
        demo_network_creation()
        demo_network_forward_pass()
        demo_ippo_with_custom_networks()
        demo_performance_comparison()
        demo_custom_user_network()
        
        print("\n\n🎉 Demo completed successfully!")
        print("\nNext steps:")
        print("1. Try different network configurations in your experiments")
        print("2. Create your own custom networks following the examples")
        print("3. Compare network performance on your specific tasks")
        print("4. Check out config/custom_networks.yaml for more examples")
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Demo interrupted by user.")
    except Exception as e:
        print(f"\n\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
