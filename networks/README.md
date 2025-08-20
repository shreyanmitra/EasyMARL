# Custom Neural Networks in EasyMARL

EasyMARL now supports flexible custom neural network architectures that work seamlessly with all algorithms. This powerful feature allows researchers to experiment with different network designs while maintaining compatibility across the entire framework.

## 🌟 Key Features

- **Built-in Networks**: 4 ready-to-use network architectures
- **Universal Compatibility**: Works with all 21+ EasyMARL algorithms  
- **MultiGrid Optimized**: Specially designed for grid-based environments
- **Easy Configuration**: Simple YAML/Python configuration
- **Custom Networks**: Import your own network architectures
- **Performance Optimized**: Efficient implementations for training and inference

## 🧠 Available Network Types

### 1. FeedForward Network (`feedforward`)
Simple multi-layer perceptron ideal for baseline experiments.

**Best for**: Simple state spaces, baseline comparisons, fast prototyping
**Features**: Configurable depth, activation functions, dropout, batch normalization

```yaml
network_type: "feedforward"
num_layers: 3
activation: "relu"
use_dropout: false
use_batch_norm: false
```

### 2. Convolutional Network (`convolutional`)
CNN architecture optimized for visual/spatial observations.

**Best for**: Grid-based environments, visual tasks, spatial reasoning
**Features**: Customizable CNN layers, pooling, feature integration

```yaml
network_type: "convolutional"
conv_layers: [32, 64, 128]
kernel_sizes: [3, 3, 3]
use_pooling: true
pool_size: 2
```

### 3. Attention Network (`attention`)
Self-attention mechanism for complex relational reasoning.

**Best for**: Long-range dependencies, multi-agent coordination, complex spatial relationships
**Features**: Multi-head attention, positional encoding, layer normalization

```yaml
network_type: "attention"
num_heads: 8
attention_dim: 128
num_attention_layers: 2
use_positional_encoding: true
```

### 4. Residual Network (`residual`)
Deep networks with skip connections for stable training.

**Best for**: Deep feature learning, complex environments, stable training
**Features**: Residual blocks, batch normalization, configurable depth

```yaml
network_type: "residual"
num_blocks: 4
block_layers: 2
use_batch_norm: true
use_dropout: true
dropout_rate: 0.1
```

## 🚀 Quick Start

### Basic Usage

```python
# Configure your algorithm with a custom network
config = {
    'network_type': 'convolutional',  # Choose network type
    'hidden_dim': 256,
    'conv_layers': [32, 64, 128],
    'learning_rate': 0.0003
}

# Run any algorithm - networks work with all algorithms!
from algorithms.model_free.policy_based.discrete_action.ippo import IPPO
algorithm = IPPO(env, config, device)
```

### YAML Configuration

```yaml
# config/my_experiment.yaml
algorithm_config:
  network_type: "attention"
  hidden_dim: 256
  num_heads: 8
  attention_dim: 128
  learning_rate: 0.0003
  gamma: 0.99
```

## 🔧 Algorithm Integration

All EasyMARL algorithms automatically support custom networks:

### IPPO with Convolutional Network
```python
config = {
    'network_type': 'convolutional',
    'conv_layers': [32, 64, 64],
    'rollout_length': 128,
    'learning_rate': 0.0003
}
ippo = IPPO(env, config, device)
```

### QMIX with Attention Network
```python
config = {
    'network_type': 'attention',
    'num_heads': 8,
    'attention_dim': 128,
    'buffer_size': 50000,
    'learning_rate': 0.0003
}
qmix = QMIX(env, config, device)
```

### MADDPG with Residual Network
```python
config = {
    'network_type': 'residual',
    'num_blocks': 3,
    'use_batch_norm': True,
    'actor_lr': 0.0001,
    'critic_lr': 0.0001
}
maddpg = MADDPG(env, config, device)
```

## 🛠️ Creating Custom Networks

You can easily create your own network architectures:

### Step 1: Create Your Network Class

```python
# my_networks.py
from networks.base_network import MultiGridCompatibleNetwork
import torch.nn as nn

class MyCustomNetwork(MultiGridCompatibleNetwork):
    """Your custom network implementation."""
    
    def __init__(self, obs_space, config, action_space, n_agents=1, agent_id=0):
        super().__init__(obs_space, config, action_space, n_agents, agent_id)
    
    def _build_network(self):
        """Build your custom architecture."""
        feature_dim = self._calculate_feature_dim()
        
        self.network = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_space)
        )
    
    def _calculate_feature_dim(self):
        """Calculate input feature dimension."""
        return (self.image_height * self.image_width * self.image_channels + 
                (1 if self.has_direction else 0))
    
    def forward(self, observations):
        """Forward pass through your network."""
        features = self._extract_features(self.process_observations(observations))
        return self.network(features)
```

### Step 2: Use Your Custom Network

```python
config = {
    'network_type': 'my_networks.MyCustomNetwork',  # module.class format
    'hidden_dim': 256,
    'learning_rate': 0.0003,
    'custom_param': 'your_value'  # Your custom parameters
}
```

## 📊 Performance Comparison

| Network Type | Parameters | Speed | Best Use Case |
|--------------|------------|-------|---------------|
| FeedForward  | ~50K       | Fastest | Simple tasks, baselines |
| Convolutional| ~200K      | Fast   | Visual environments |
| Attention    | ~150K      | Medium | Complex reasoning |
| Residual     | ~300K      | Medium | Deep learning |

## 🎯 Environment-Specific Recommendations

### Cooperative Environments
- **Recommended**: Attention networks for coordination
- **Configuration**: High attention heads, positional encoding
```yaml
network_type: "attention"
num_heads: 8
attention_dim: 128
use_positional_encoding: true
```

### Competitive Environments  
- **Recommended**: Convolutional networks for spatial awareness
- **Configuration**: Multiple conv layers, pooling
```yaml
network_type: "convolutional"
conv_layers: [64, 128, 256]
use_pooling: true
```

### Large-Scale Environments
- **Recommended**: FeedForward networks for efficiency
- **Configuration**: Moderate depth, no unnecessary features
```yaml
network_type: "feedforward"
num_layers: 2
hidden_dim: 128
```

## 🔬 Advanced Features

### Network Factory
Access networks programmatically:

```python
from networks.base_network import NetworkFactory

network = NetworkFactory.create_network(
    network_type='convolutional',
    obs_space={'image': (7, 7, 3), 'direction': 4},
    config={'conv_layers': [32, 64]},
    action_space=6
)
```

### Observation Processing
All networks handle MultiGrid observations automatically:

```python
observations = {
    'image': torch.tensor(...),    # Grid state
    'direction': torch.tensor(...) # Agent direction
}

output = network(observations)  # Automatic processing
```

### Configuration Inheritance
Networks inherit from base classes with consistent interfaces:

```python
# All networks support these methods
network.forward(observations)              # Forward pass
network.process_observations(obs)          # Preprocessing
network._extract_features(processed_obs)   # Feature extraction
```

## 📁 File Structure

```
networks/
├── base_network.py          # Base classes and interfaces
├── custom_networks.py       # Built-in network implementations
└── multigrid_network.py    # Legacy network (still supported)

config/
└── custom_networks.yaml    # Example configurations

examples/
└── demo_custom_networks.py # Demonstration script
```

## 🚨 Common Issues and Solutions

### Issue: Network Import Error
```
Could not import network type 'my_network'
```
**Solution**: Ensure your network file is in Python path and class inherits from `BaseNetwork`.

### Issue: Dimension Mismatch
```
Size mismatch in linear layer
```
**Solution**: Check `_calculate_feature_dim()` method in your custom network.

### Issue: Slow Training
```
Training is slower than expected
```
**Solution**: Use lighter networks like `feedforward` or reduce hidden dimensions.

## 🔍 Examples and Tutorials

### Complete Example Script
Run the comprehensive demo:
```bash
python examples/demo_custom_networks.py
```

### Configuration Examples
Check out pre-configured setups:
```bash
cat config/custom_networks.yaml
```

### Algorithm-Specific Examples
See how different algorithms use custom networks:
- `examples/ippo_with_custom_networks.py`
- `examples/qmix_attention_demo.py`
- `examples/maddpg_residual_demo.py`

## 🤝 Contributing

Want to add a new network type? Follow these steps:

1. **Create Network Class**: Inherit from `MultiGridCompatibleNetwork`
2. **Implement Required Methods**: `_build_network()`, `forward()`
3. **Add to Factory**: Update `NetworkFactory.create_network()`
4. **Add Tests**: Create unit tests for your network
5. **Update Documentation**: Add to this README

## 📚 Research Applications

Custom networks enable cutting-edge research:

- **Architecture Search**: Experiment with novel designs
- **Transfer Learning**: Pre-trained network components  
- **Multi-Task Learning**: Shared network backbones
- **Meta-Learning**: Adaptive network structures
- **Continual Learning**: Networks that grow over time

## 🏆 Success Stories

Researchers have successfully used custom networks for:

- **Robot Swarms**: Attention networks for coordination
- **Traffic Control**: CNNs for spatial traffic patterns  
- **Game AI**: Residual networks for complex strategies
- **Resource Allocation**: Custom networks for optimization

## 📧 Support

Need help with custom networks?

- **Documentation**: Check this README and code comments
- **Examples**: Run `examples/demo_custom_networks.py`
- **Issues**: Open GitHub issues for bugs or questions
- **Discussions**: Join community discussions for research ideas

---

**Ready to experiment with custom networks? Start with the demo script and configuration examples above!** 🚀
