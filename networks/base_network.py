"""
Base Network Interface for EasyMARL Custom Networks

This module provides the foundation for creating custom neural networks that work
with all EasyMARL algorithms and are compatible with MultiGrid environments.

Key Features:
- Abstract base class ensuring consistent interface across all networks
- MultiGrid environment compatibility built-in
- Support for both discrete and continuous action spaces
- Flexible observation processing for different input types
- Integration with all MARL algorithms (QMIX, VDN, IPPO, MADDPG, etc.)

For Users:
Create your custom network by inheriting from BaseNetwork and implementing
the required methods. Your network will automatically work with all algorithms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Dict, Union, Tuple, Any, Optional
import numpy as np


class BaseNetwork(nn.Module, ABC):
    """
    Abstract base class for all custom networks in EasyMARL.
    
    This ensures consistent interface across all algorithms while providing
    flexibility for users to implement their own architectures.
    
    Key Requirements:
    1. Must handle MultiGrid observation format
    2. Must support both value-based and policy-based algorithms
    3. Must be compatible with discrete and continuous action spaces
    4. Must provide consistent output format for all algorithms
    
    For Algorithm Developers:
    All algorithms can use any network that inherits from this base class,
    ensuring modularity and flexibility in the framework.
    """
    
    def __init__(self, obs_space: Dict, config: Dict, action_space: int, 
                 n_agents: int = 1, agent_id: int = 0):
        """
        Initialize the base network.
        
        Args:
            obs_space (Dict): Observation space specification
                             Example: {'image': (7, 7, 3), 'direction': 4}
            config (Dict): Configuration parameters
            action_space (int): Number of available actions
            n_agents (int): Total number of agents (for parameter sharing)
            agent_id (int): ID of this specific agent
        """
        super().__init__()
        
        self.obs_space = obs_space
        self.config = config
        self.action_space = action_space
        self.n_agents = n_agents
        self.agent_id = agent_id
        
        # Device management
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Network configuration
        self.hidden_dim = config.get('hidden_dim', 128)
        self.use_parameter_sharing = config.get('parameter_sharing', False)
        
        # Initialize network components
        self._build_network()
        
    @abstractmethod
    def _build_network(self):
        """
        Build the network architecture.
        
        This method must be implemented by all custom networks.
        Define your layers, activations, and any special components here.
        """
        pass
    
    @abstractmethod
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            observations (Dict[str, torch.Tensor]): Processed observations
                Expected format for MultiGrid:
                {
                    'image': tensor of shape (batch_size, height, width, channels),
                    'direction': tensor of shape (batch_size, 1)
                }
        
        Returns:
            torch.Tensor: Network output
                For value-based algorithms: Q-values for each action
                For policy-based algorithms: Action logits or mean/std
                Shape: (batch_size, output_dim)
        """
        pass
    
    def process_observations(self, observations: Union[Dict, np.ndarray, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Process raw observations into tensor format suitable for the network.
        
        This method handles the conversion of MultiGrid observations into the
        format expected by neural networks, ensuring compatibility across
        all algorithms.
        
        Args:
            observations: Raw observations from environment
                         Can be dict, numpy array, or tensor
        
        Returns:
            Dict[str, torch.Tensor]: Processed observations ready for network
        """
        if isinstance(observations, dict):
            processed = {}
            
            for key, value in observations.items():
                if isinstance(value, np.ndarray):
                    processed[key] = torch.tensor(value, dtype=torch.float32).to(self.device)
                elif isinstance(value, torch.Tensor):
                    processed[key] = value.to(self.device)
                elif isinstance(value, (int, float)):
                    processed[key] = torch.tensor([value], dtype=torch.float32).to(self.device)
                else:
                    # Handle lists or other formats
                    processed[key] = torch.tensor(value, dtype=torch.float32).to(self.device)
                
                # Ensure batch dimension
                if len(processed[key].shape) == 0:
                    processed[key] = processed[key].unsqueeze(0)
                elif len(processed[key].shape) == 1 and key != 'direction':
                    processed[key] = processed[key].unsqueeze(0)
                elif len(processed[key].shape) == 3 and key == 'image':
                    processed[key] = processed[key].unsqueeze(0)
            
            return processed
        
        else:
            # Handle non-dict observations (convert to standard format)
            if isinstance(observations, np.ndarray):
                tensor = torch.tensor(observations, dtype=torch.float32).to(self.device)
            elif isinstance(observations, torch.Tensor):
                tensor = observations.to(self.device)
            else:
                tensor = torch.tensor([observations], dtype=torch.float32).to(self.device)
            
            # Ensure batch dimension
            if len(tensor.shape) == 1:
                tensor = tensor.unsqueeze(0)
            
            return {'features': tensor}
    
    def get_output_dim(self) -> int:
        """
        Get the output dimension of the network.
        
        Returns:
            int: Output dimension (e.g., action_space for Q-networks)
        """
        return self.action_space
    
    def get_feature_dim(self) -> int:
        """
        Get the feature dimension after observation processing.
        
        Returns:
            int: Feature dimension for building subsequent layers
        """
        # Create a dummy observation to get feature dimensions
        dummy_obs = self._create_dummy_observation()
        with torch.no_grad():
            features = self._extract_features(dummy_obs)
            return features.shape[-1]
    
    def _create_dummy_observation(self) -> Dict[str, torch.Tensor]:
        """
        Create a dummy observation for dimension calculation.
        
        Returns:
            Dict[str, torch.Tensor]: Dummy observation matching obs_space
        """
        dummy_obs = {}
        
        for key, shape in self.obs_space.items():
            if key == 'image':
                if isinstance(shape, (list, tuple)) and len(shape) == 3:
                    dummy_obs[key] = torch.zeros(1, *shape).to(self.device)
                else:
                    dummy_obs[key] = torch.zeros(1, 7, 7, 3).to(self.device)  # Default MultiGrid size
            elif key == 'direction':
                dummy_obs[key] = torch.zeros(1, 1).to(self.device)
            else:
                if isinstance(shape, (int, float)):
                    dummy_obs[key] = torch.zeros(1, 1).to(self.device)
                elif isinstance(shape, (list, tuple)):
                    dummy_obs[key] = torch.zeros(1, *shape).to(self.device)
                else:
                    dummy_obs[key] = torch.zeros(1, shape).to(self.device)
        
        return dummy_obs
    
    @abstractmethod
    def _extract_features(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Extract features from processed observations.
        
        This method should implement the feature extraction part of your network,
        which can then be used by different heads (policy, value, Q-function).
        
        Args:
            observations (Dict[str, torch.Tensor]): Processed observations
        
        Returns:
            torch.Tensor: Extracted features (batch_size, feature_dim)
        """
        pass


class MultiGridCompatibleNetwork(BaseNetwork):
    """
    A base class specifically designed for MultiGrid environment compatibility.
    
    This class provides common patterns for processing MultiGrid observations
    and can be used as a starting point for custom networks.
    
    MultiGrid Observation Format:
    - 'image': (height, width, channels) - Visual representation of the grid
    - 'direction': (1,) - Agent's current facing direction
    
    This class handles the standard preprocessing and provides hooks for
    custom feature extraction and output processing.
    """
    
    def __init__(self, obs_space: Dict, config: Dict, action_space: int, 
                 n_agents: int = 1, agent_id: int = 0):
        """Initialize MultiGrid-compatible network."""
        super().__init__(obs_space, config, action_space, n_agents, agent_id)
        
        # Extract MultiGrid-specific dimensions
        self.image_shape = obs_space.get('image', (7, 7, 3))
        self.has_direction = 'direction' in obs_space
        self.direction_dim = obs_space.get('direction', 4) if self.has_direction else 0
        
        # Calculate image feature dimensions
        if isinstance(self.image_shape, (list, tuple)) and len(self.image_shape) == 3:
            self.image_height, self.image_width, self.image_channels = self.image_shape
        else:
            self.image_height, self.image_width, self.image_channels = 7, 7, 3
        
    def _extract_features(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Extract features from MultiGrid observations.
        
        This default implementation provides a reasonable feature extraction
        for MultiGrid environments. Override this for custom feature processing.
        
        Args:
            observations (Dict[str, torch.Tensor]): Processed observations
        
        Returns:
            torch.Tensor: Extracted features
        """
        features = []
        
        # Process image observation
        if 'image' in observations:
            image = observations['image']
            # Flatten image or apply convolution (override for custom processing)
            image_features = image.flatten(start_dim=1)
            features.append(image_features)
        
        # Process direction observation
        if 'direction' in observations:
            direction = observations['direction']
            if len(direction.shape) == 1:
                direction = direction.unsqueeze(1)
            features.append(direction.float())
        
        # Process any additional observations
        for key, value in observations.items():
            if key not in ['image', 'direction']:
                if len(value.shape) == 1:
                    value = value.unsqueeze(1)
                features.append(value.float())
        
        # Concatenate all features
        if features:
            return torch.cat(features, dim=1)
        else:
            # Fallback if no features found
            batch_size = next(iter(observations.values())).shape[0]
            return torch.zeros(batch_size, self.hidden_dim).to(self.device)


class NetworkFactory:
    """
    Factory class for creating custom networks.
    
    This factory allows algorithms to create networks dynamically based on
    user configuration while maintaining compatibility with all algorithms.
    """
    
    @staticmethod
    def create_network(network_type: str, obs_space: Dict, config: Dict, 
                      action_space: int, n_agents: int = 1, agent_id: int = 0) -> BaseNetwork:
        """
        Create a network instance based on the specified type.
        
        Args:
            network_type (str): Type of network to create
            obs_space (Dict): Observation space specification
            config (Dict): Configuration parameters
            action_space (int): Number of available actions
            n_agents (int): Total number of agents
            agent_id (int): ID of this specific agent
        
        Returns:
            BaseNetwork: Instantiated network
        """
        # Built-in network types that work with MultiGrid
        network_map = {
            'feedforward': 'FeedForwardNetwork',
            'convolutional': 'ConvolutionalNetwork',
            'attention': 'AttentionNetwork', 
            'residual': 'ResidualNetwork',
            'multigrid': 'MultiGridCompatibleNetwork'
        }
        
        if network_type.lower() in network_map:
            if network_type.lower() == 'multigrid':
                # Use the local MultiGridCompatibleNetwork class
                return MultiGridCompatibleNetwork(obs_space, config, action_space, n_agents, agent_id)
            else:
                # Import built-in custom networks
                try:
                    from .custom_networks import (
                        FeedForwardNetwork, ConvolutionalNetwork, 
                        AttentionNetwork, ResidualNetwork
                    )
                    
                    network_classes = {
                        'feedforward': FeedForwardNetwork,
                        'convolutional': ConvolutionalNetwork,
                        'attention': AttentionNetwork,
                        'residual': ResidualNetwork
                    }
                    
                    network_class = network_classes[network_type.lower()]
                    return network_class(obs_space, config, action_space, n_agents, agent_id)
                    
                except ImportError as e:
                    print(f"Warning: Could not import built-in network '{network_type}': {e}")
                    print("Falling back to MultiGridCompatibleNetwork")
                    return MultiGridCompatibleNetwork(obs_space, config, action_space, n_agents, agent_id)
        else:
            # Try to import custom user network
            try:
                # Allow users to specify custom network classes
                if '.' in network_type:
                    module_name, class_name = network_type.rsplit('.', 1)
                    module = __import__(module_name, fromlist=[class_name])
                    network_class = getattr(module, class_name)
                    return network_class(obs_space, config, action_space, n_agents, agent_id)
                else:
                    raise ValueError(f"Unknown network type: {network_type}")
            except (ImportError, AttributeError) as e:
                print(f"Warning: Could not create network '{network_type}': {e}")
                print("Falling back to MultiGridCompatibleNetwork")
                return MultiGridCompatibleNetwork(obs_space, config, action_space, n_agents, agent_id)


# Utility function for algorithms to get networks
def get_network(obs_space: Dict, config: Dict, action_space: int, 
               n_agents: int = 1, agent_id: int = 0) -> BaseNetwork:
    """
    Convenience function for algorithms to get the appropriate network.
    
    Args:
        obs_space (Dict): Observation space specification
        config (Dict): Configuration parameters (should include 'network_type')
        action_space (int): Number of available actions
        n_agents (int): Total number of agents
        agent_id (int): ID of this specific agent
    
    Returns:
        BaseNetwork: Network instance ready for use
    """
    network_type = config.get('network_type', 'feedforward')
    
    return NetworkFactory.create_network(
        network_type, obs_space, config, action_space, n_agents, agent_id
    )
