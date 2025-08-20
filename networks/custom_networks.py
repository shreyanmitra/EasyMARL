"""
Built-in Custom Networks for EasyMARL

This module provides a collection of pre-built custom networks that users can
choose from when configuring their MARL algorithms. All networks are compatible
with MultiGrid environments and work with all EasyMARL algorithms.

Available Networks:
1. FeedForwardNetwork - Simple fully connected layers
2. ConvolutionalNetwork - CNN for visual observations  
3. AttentionNetwork - Self-attention mechanism
4. ResidualNetwork - Skip connections for deep networks

Usage:
Configure your algorithm to use any of these networks by setting the 
'network_type' parameter in your configuration.

Example:
config = {
    'network_type': 'convolutional',
    'hidden_dim': 256,
    'conv_layers': [32, 64, 128]
}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
import math

from .base_network import BaseNetwork, MultiGridCompatibleNetwork


class FeedForwardNetwork(MultiGridCompatibleNetwork):
    """
    Simple Feed Forward Neural Network.
    
    A standard multi-layer perceptron with customizable depth and width.
    Perfect for environments with simple observation spaces or as a baseline.
    
    Features:
    - Configurable number of layers and hidden dimensions
    - Choice of activation functions
    - Optional dropout for regularization
    - Batch normalization support
    
    Best for: Simple state representations, baseline experiments
    """
    
    def __init__(self, obs_space: Dict, config: Dict, action_space: int, 
                 n_agents: int = 1, agent_id: int = 0):
        """Initialize FeedForward network with proper signature."""
        super().__init__(obs_space, config, action_space, n_agents, agent_id)
    
    def _build_network(self):
        """Build the feed-forward network architecture."""
        # Network configuration
        self.num_layers = self.config.get('num_layers', 3)
        self.activation = self.config.get('activation', 'relu')
        self.use_dropout = self.config.get('use_dropout', False)
        self.dropout_rate = self.config.get('dropout_rate', 0.1)
        self.use_batch_norm = self.config.get('use_batch_norm', False)
        
        # Get feature dimension
        feature_dim = self._calculate_feature_dim()
        
        # Build layers
        layers = []
        input_dim = feature_dim
        
        for i in range(self.num_layers - 1):
            # Linear layer
            layers.append(nn.Linear(input_dim, self.hidden_dim))
            
            # Batch normalization
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(self.hidden_dim))
            
            # Activation
            layers.append(self._get_activation())
            
            # Dropout
            if self.use_dropout:
                layers.append(nn.Dropout(self.dropout_rate))
            
            input_dim = self.hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, self.action_space))
        
        self.network = nn.Sequential(*layers)
    
    def _calculate_feature_dim(self) -> int:
        """Calculate the total feature dimension."""
        feature_dim = 0
        
        # Image features (flattened)
        feature_dim += self.image_height * self.image_width * self.image_channels
        
        # Direction features
        if self.has_direction:
            feature_dim += 1
        
        # Additional features
        for key, shape in self.obs_space.items():
            if key not in ['image', 'direction']:
                if isinstance(shape, (int, float)):
                    feature_dim += 1
                elif isinstance(shape, (list, tuple)):
                    feature_dim += int(torch.prod(torch.tensor(shape)))
        
        return feature_dim
    
    def _get_activation(self):
        """Get activation function based on configuration."""
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU(),
            'gelu': nn.GELU()
        }
        return activations.get(self.activation, nn.ReLU())
    
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass through the feed-forward network."""
        # Process observations
        processed_obs = self.process_observations(observations)
        
        # Extract features
        features = self._extract_features(processed_obs)
        
        # Forward through network
        output = self.network(features)
        
        return output


class ConvolutionalNetwork(MultiGridCompatibleNetwork):
    """
    Convolutional Neural Network for visual observations.
    
    Designed specifically for grid-based environments like MultiGrid.
    Uses convolution layers to process spatial information efficiently.
    
    Features:
    - Customizable CNN architecture
    - Spatial feature extraction
    - Integration with non-visual features
    - Efficient processing of grid-based observations
    
    Best for: Visual environments, grid-based tasks, spatial reasoning
    """
    
    def __init__(self, obs_space: Dict, config: Dict, action_space: int, 
                 n_agents: int = 1, agent_id: int = 0):
        """Initialize Convolutional network with proper signature."""
        super().__init__(obs_space, config, action_space, n_agents, agent_id)
    
    def _build_network(self):
        """Build the convolutional network architecture."""
        # CNN configuration
        self.conv_layers = self.config.get('conv_layers', [32, 64, 64])
        self.kernel_sizes = self.config.get('kernel_sizes', [3, 3, 3])
        self.strides = self.config.get('strides', [1, 1, 1])
        self.use_pooling = self.config.get('use_pooling', True)
        self.pool_size = self.config.get('pool_size', 2)
        
        # Ensure lists are same length
        num_conv_layers = len(self.conv_layers)
        if len(self.kernel_sizes) != num_conv_layers:
            self.kernel_sizes = [self.kernel_sizes[0]] * num_conv_layers
        if len(self.strides) != num_conv_layers:
            self.strides = [self.strides[0]] * num_conv_layers
        
        # Build convolutional layers
        conv_layers = []
        in_channels = self.image_channels
        
        for i, out_channels in enumerate(self.conv_layers):
            conv_layers.append(nn.Conv2d(
                in_channels, out_channels, 
                kernel_size=self.kernel_sizes[i],
                stride=self.strides[i],
                padding=self.kernel_sizes[i] // 2
            ))
            conv_layers.append(nn.ReLU())
            
            if self.use_pooling and i < num_conv_layers - 1:
                conv_layers.append(nn.MaxPool2d(self.pool_size))
            
            in_channels = out_channels
        
        self.conv_net = nn.Sequential(*conv_layers)
        
        # Calculate conv output dimension
        conv_output_dim = self._calculate_conv_output_dim()
        
        # Add non-visual features
        non_visual_dim = 0
        if self.has_direction:
            non_visual_dim += 1
        
        # Additional features
        for key, shape in self.obs_space.items():
            if key not in ['image', 'direction']:
                if isinstance(shape, (int, float)):
                    non_visual_dim += 1
                elif isinstance(shape, (list, tuple)):
                    non_visual_dim += int(torch.prod(torch.tensor(shape)))
        
        # Build final layers
        total_features = conv_output_dim + non_visual_dim
        
        self.fc_layers = nn.Sequential(
            nn.Linear(total_features, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_space)
        )
    
    def _calculate_conv_output_dim(self) -> int:
        """Calculate the output dimension after convolution layers."""
        # Create dummy input
        dummy_input = torch.zeros(1, self.image_channels, self.image_height, self.image_width)
        
        with torch.no_grad():
            conv_output = self.conv_net(dummy_input)
            return int(torch.prod(torch.tensor(conv_output.shape[1:])))
    
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass through the convolutional network."""
        # Process observations
        processed_obs = self.process_observations(observations)
        
        features = []
        
        # Process image with CNN
        if 'image' in processed_obs:
            image = processed_obs['image']
            # Ensure correct format: (batch, channels, height, width)
            if len(image.shape) == 4 and image.shape[-1] == self.image_channels:
                image = image.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
            
            conv_features = self.conv_net(image)
            conv_features = conv_features.flatten(start_dim=1)
            features.append(conv_features)
        
        # Add non-visual features
        for key, value in processed_obs.items():
            if key != 'image':
                if len(value.shape) == 1:
                    value = value.unsqueeze(1)
                features.append(value.float())
        
        # Concatenate all features
        if features:
            combined_features = torch.cat(features, dim=1)
        else:
            batch_size = next(iter(processed_obs.values())).shape[0]
            combined_features = torch.zeros(batch_size, self.hidden_dim).to(self.device)
        
        # Forward through fully connected layers
        output = self.fc_layers(combined_features)
        
        return output


class AttentionNetwork(MultiGridCompatibleNetwork):
    """
    Self-Attention Network for processing sequential or spatial information.
    
    Uses multi-head self-attention to capture relationships between different
    parts of the observation space. Particularly useful for environments
    requiring long-range dependencies or complex spatial reasoning.
    
    Features:
    - Multi-head self-attention mechanism
    - Positional encoding for spatial information
    - Layer normalization and residual connections
    - Configurable attention heads and dimensions
    
    Best for: Complex spatial reasoning, long-range dependencies, relational tasks
    """
    
    def __init__(self, obs_space: Dict, config: Dict, action_space: int, 
                 n_agents: int = 1, agent_id: int = 0):
        """Initialize Attention network with proper signature."""
        super().__init__(obs_space, config, action_space, n_agents, agent_id)
    
    def _build_network(self):
        """Build the attention-based network architecture."""
        # Attention configuration
        self.num_heads = self.config.get('num_heads', 8)
        self.attention_dim = self.config.get('attention_dim', 128)
        self.num_attention_layers = self.config.get('num_attention_layers', 2)
        self.use_positional_encoding = self.config.get('use_positional_encoding', True)
        
        # Ensure attention_dim is divisible by num_heads
        self.attention_dim = (self.attention_dim // self.num_heads) * self.num_heads
        
        # Input projection
        input_dim = self.image_height * self.image_width * self.image_channels
        if self.has_direction:
            input_dim += 1
        
        self.input_projection = nn.Linear(input_dim, self.attention_dim)
        
        # Positional encoding for spatial information
        if self.use_positional_encoding:
            self.pos_encoding = PositionalEncoding(self.attention_dim)
        
        # Multi-head attention layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=self.attention_dim,
                num_heads=self.num_heads,
                batch_first=True
            ) for _ in range(self.num_attention_layers)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.attention_dim) for _ in range(self.num_attention_layers)
        ])
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(self.attention_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_space)
        )
    
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass through the attention network."""
        # Process observations
        processed_obs = self.process_observations(observations)
        
        # Extract and flatten features
        features = self._extract_features(processed_obs)
        
        # Project to attention dimension
        x = self.input_projection(features)
        
        # Add positional encoding if enabled
        if self.use_positional_encoding:
            x = self.pos_encoding(x.unsqueeze(1))  # Add sequence dimension
        else:
            x = x.unsqueeze(1)  # Add sequence dimension
        
        # Apply attention layers
        for attention, layer_norm in zip(self.attention_layers, self.layer_norms):
            # Self-attention with residual connection
            attended, _ = attention(x, x, x)
            x = layer_norm(x + attended)
        
        # Global average pooling over sequence dimension
        x = x.mean(dim=1)
        
        # Output projection
        output = self.output_layers(x)
        
        return output


class ResidualNetwork(MultiGridCompatibleNetwork):
    """
    Residual Network with skip connections.
    
    Uses residual connections to enable training of deeper networks.
    Helps with gradient flow and allows for more complex feature learning
    while maintaining training stability.
    
    Features:
    - Residual blocks with skip connections
    - Configurable depth and width
    - Batch normalization for training stability
    - Optional dropout for regularization
    
    Best for: Deep feature learning, complex environments, stable training
    """
    
    def __init__(self, obs_space: Dict, config: Dict, action_space: int, 
                 n_agents: int = 1, agent_id: int = 0):
        """Initialize Residual network with proper signature."""
        super().__init__(obs_space, config, action_space, n_agents, agent_id)
    
    def _build_network(self):
        """Build the residual network architecture."""
        # Residual network configuration
        self.num_blocks = self.config.get('num_blocks', 3)
        self.block_layers = self.config.get('block_layers', 2)
        self.use_batch_norm = self.config.get('use_batch_norm', True)
        self.use_dropout = self.config.get('use_dropout', False)
        self.dropout_rate = self.config.get('dropout_rate', 0.1)
        
        # Input processing
        feature_dim = self._calculate_feature_dim()
        
        # Input projection to hidden dimension
        self.input_projection = nn.Linear(feature_dim, self.hidden_dim)
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(
                self.hidden_dim, 
                self.block_layers,
                self.use_batch_norm,
                self.use_dropout,
                self.dropout_rate
            ) for _ in range(self.num_blocks)
        ])
        
        # Output layer
        self.output_layer = nn.Linear(self.hidden_dim, self.action_space)
    
    def _calculate_feature_dim(self) -> int:
        """Calculate the total feature dimension."""
        feature_dim = 0
        
        # Image features (flattened)
        feature_dim += self.image_height * self.image_width * self.image_channels
        
        # Direction features
        if self.has_direction:
            feature_dim += 1
        
        # Additional features
        for key, shape in self.obs_space.items():
            if key not in ['image', 'direction']:
                if isinstance(shape, (int, float)):
                    feature_dim += 1
                elif isinstance(shape, (list, tuple)):
                    feature_dim += int(torch.prod(torch.tensor(shape)))
        
        return feature_dim
    
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass through the residual network."""
        # Process observations
        processed_obs = self.process_observations(observations)
        
        # Extract features
        features = self._extract_features(processed_obs)
        
        # Input projection
        x = F.relu(self.input_projection(features))
        
        # Apply residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Output projection
        output = self.output_layer(x)
        
        return output


class ResidualBlock(nn.Module):
    """
    A single residual block with skip connection.
    """
    
    def __init__(self, hidden_dim: int, num_layers: int = 2, 
                 use_batch_norm: bool = True, use_dropout: bool = False,
                 dropout_rate: float = 0.1):
        super().__init__()
        
        layers = []
        
        for i in range(num_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            if i < num_layers - 1:  # No activation on last layer
                layers.append(nn.ReLU())
            
            if use_dropout:
                layers.append(nn.Dropout(dropout_rate))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass with residual connection."""
        return F.relu(x + self.layers(x))


class PositionalEncoding(nn.Module):
    """
    Positional encoding for attention networks.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        """Add positional encoding to input."""
        return x + self.pe[:, :x.size(1)]
