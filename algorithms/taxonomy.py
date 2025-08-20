"""
EasyMARL Algorithm Taxonomy System

This module implements a comprehensive taxonomy for organizing reinforcement learning
algorithms based on their fundamental characteristics and methodologies.

The taxonomy follows the structure outlined in modern RL literature and provides:
1. Clear categorization of algorithms by their core principles
2. Educational progression paths for learners
3. Systematic comparison framework for researchers
4. Intuitive organization for practical applications

Created: August 17, 2025
Authors: EasyMARL Development Team
License: MIT License

Copyright (c) 2025 EasyMARL. All rights reserved.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Type, Optional
import importlib


class AlgorithmCategory(Enum):
    """
    Primary categorization of RL algorithms based on model usage.
    
    This is the highest-level distinction in our taxonomy, determining
    whether an algorithm uses environment models for planning.
    """
    MODEL_FREE = "model_free"      # Learn directly from experience
    MODEL_BASED = "model_based"    # Use environment models for planning


class ModelFreeCategory(Enum):
    """
    Subcategories within model-free reinforcement learning.
    
    These categories distinguish how algorithms approach learning
    without explicit environment models.
    """
    VALUE_BASED = "value_based"      # Learn value functions
    POLICY_BASED = "policy_based"    # Learn policies directly
    ACTOR_CRITIC = "actor_critic"    # Combine value and policy learning


class ValueBasedCategory(Enum):
    """
    Subcategories within value-based methods.
    
    Distinguishes how value functions are represented and learned.
    """
    TABULAR = "tabular"           # Table-based value storage
    APPROXIMATION = "approximation"  # Function approximation (neural networks)


class TabularCategory(Enum):
    """
    Subcategories within tabular methods.
    
    Distinguishes whether state information is considered.
    """
    STATELESS = "stateless"       # Multi-armed bandits
    STATEFUL = "stateful"         # Traditional tabular RL


class PolicyBasedCategory(Enum):
    """
    Subcategories within policy-based methods.
    
    Distinguishes based on action space characteristics.
    """
    DISCRETE_ACTION = "discrete_action"      # Discrete action spaces
    CONTINUOUS_ACTION = "continuous_action"  # Continuous action spaces


class ModelBasedCategory(Enum):
    """
    Subcategories within model-based methods.
    
    Distinguishes how environment models are obtained.
    """
    GIVEN_MODEL = "given_model"      # Predefined models (e.g., chess)
    LEARNED_MODEL = "learned_model"  # Models learned from data


class OptimizationMethod(Enum):
    """
    Methods for policy optimization in model-based approaches.
    
    Distinguishes how policies are improved using models.
    """
    PLANNING_BASED = "planning_based"    # Search-based methods (MCTS)
    GRADIENT_BASED = "gradient_based"    # Gradient optimization


class AlgorithmTaxonomy:
    """
    Comprehensive taxonomy system for organizing RL algorithms.
    
    This class provides a structured way to categorize, discover, and
    instantiate RL algorithms based on their fundamental characteristics.
    
    Educational Benefits:
    - Clear learning progression from simple to complex
    - Understanding of algorithm relationships
    - Systematic comparison framework
    
    Research Benefits:
    - Identification of algorithm gaps
    - Systematic evaluation protocols
    - Clear communication of contributions
    
    Practical Benefits:
    - Algorithm selection guidance
    - Intuitive API organization
    - Maintainable code structure
    """
    
    def __init__(self):
        """Initialize the taxonomy with algorithm mappings."""
        self._algorithm_registry = self._build_algorithm_registry()
    
    def _build_algorithm_registry(self) -> Dict:
        """
        Build the complete algorithm registry based on taxonomy.
        
        This creates a hierarchical mapping from taxonomy categories
        to specific algorithm implementations.
        
        Returns:
            Dict: Nested dictionary mapping categories to algorithms
        """
        return {
            AlgorithmCategory.MODEL_FREE: {
                ModelFreeCategory.VALUE_BASED: {
                    ValueBasedCategory.APPROXIMATION: [
                        'qmix', 'vdn', 'qtran', 'iql', 'mfq'
                    ],
                    ValueBasedCategory.TABULAR: {
                        TabularCategory.STATEFUL: [
                            'nashq', 'minimaxq', 'wolfphc', 'hql', 'lql'
                        ],
                        TabularCategory.STATELESS: [
                            # Multi-armed bandit algorithms would go here
                        ]
                    }
                },
                ModelFreeCategory.POLICY_BASED: {
                    PolicyBasedCategory.DISCRETE_ACTION: [
                        'ippo', 'mappo'
                    ]
                    # Note: Removed continuous_action category since MultiGrid uses discrete actions
                    # MADDPG moved to actor_critic as it's fundamentally an actor-critic algorithm
                },
                ModelFreeCategory.ACTOR_CRITIC: [
                    'coma', 'comacomm', 'maacc', 'maven', 'dcg', 'nfsp', 
                    'maddpg', 'maddpgcomm'  # Moved from policy_based/continuous_action
                ]
            },
            AlgorithmCategory.MODEL_BASED: {
                ModelBasedCategory.GIVEN_MODEL: {
                    OptimizationMethod.PLANNING_BASED: [
                        # AlphaZero-style algorithms would go here
                    ]
                },
                ModelBasedCategory.LEARNED_MODEL: {
                    OptimizationMethod.GRADIENT_BASED: [
                        # Dreamer-style algorithms would go here
                    ],
                    OptimizationMethod.PLANNING_BASED: [
                        # MuZero-style algorithms would go here
                    ]
                }
            }
        }
    
    def get_algorithms_by_category(
        self, 
        category: AlgorithmCategory,
        subcategory: Optional[Enum] = None
    ) -> List[str]:
        """
        Retrieve algorithms belonging to a specific category.
        
        Args:
            category: Primary algorithm category
            subcategory: Optional subcategory for filtering
            
        Returns:
            List[str]: Algorithm names in the specified category
        """
        algorithms = []
        category_data = self._algorithm_registry.get(category, {})
        
        if subcategory is None:
            # Return all algorithms in category
            algorithms = self._extract_all_algorithms(category_data)
        else:
            # Return algorithms in specific subcategory
            subcategory_data = category_data.get(subcategory, {})
            algorithms = self._extract_all_algorithms(subcategory_data)
        
        return algorithms
    
    def _extract_all_algorithms(self, data) -> List[str]:
        """Recursively extract all algorithm names from nested structure."""
        algorithms = []
        
        if isinstance(data, list):
            algorithms.extend(data)
        elif isinstance(data, dict):
            for value in data.values():
                algorithms.extend(self._extract_all_algorithms(value))
        
        return algorithms
    
    def get_algorithm_path(self, algorithm_name: str) -> Optional[str]:
        """
        Get the taxonomic path for a specific algorithm.
        
        Args:
            algorithm_name: Name of the algorithm
            
        Returns:
            Optional[str]: Path to algorithm in taxonomy, or None if not found
        """
        path = self._find_algorithm_path(algorithm_name, self._algorithm_registry, [])
        return " → ".join(path) if path else None
    
    def _find_algorithm_path(self, algorithm_name: str, data, current_path: List[str]) -> Optional[List[str]]:
        """Recursively find the path to an algorithm in the taxonomy."""
        if isinstance(data, list):
            if algorithm_name in data:
                return current_path
        elif isinstance(data, dict):
            for key, value in data.items():
                new_path = current_path + [key.value if hasattr(key, 'value') else str(key)]
                result = self._find_algorithm_path(algorithm_name, value, new_path)
                if result:
                    return result
        return None
    
    def get_learning_progression(self) -> List[Dict[str, any]]:
        """
        Get recommended learning progression through algorithms.
        
        Returns a structured path from simple to complex algorithms,
        designed for educational purposes.
        
        Returns:
            List[Dict]: Learning progression with explanations
        """
        return [
            {
                "level": "Beginner",
                "category": "Tabular Value-Based",
                "algorithms": ["nashq", "minimaxq"],
                "description": "Start with simple tabular methods to understand basic RL concepts",
                "concepts": ["Value functions", "Bellman equations", "Exploration vs exploitation"]
            },
            {
                "level": "Intermediate",
                "category": "Approximation Value-Based",
                "algorithms": ["vdn", "qmix"],
                "description": "Learn function approximation and value decomposition",
                "concepts": ["Neural networks", "Value decomposition", "Credit assignment"]
            },
            {
                "level": "Advanced",
                "category": "Policy-Based",
                "algorithms": ["ippo", "mappo"],
                "description": "Understand direct policy optimization",
                "concepts": ["Policy gradients", "Trust regions", "Variance reduction"]
            },
            {
                "level": "Expert",
                "category": "Actor-Critic",
                "algorithms": ["coma", "maacc"],
                "description": "Master sophisticated multi-agent coordination",
                "concepts": ["Centralized training", "Decentralized execution", "Communication"]
            }
        ]
    
    def get_algorithm_recommendations(
        self, 
        environment_type: str,
        experience_level: str,
        performance_priority: str
    ) -> List[str]:
        """
        Recommend algorithms based on user requirements.
        
        Args:
            environment_type: Type of environment ("cooperative", "competitive", "mixed")
            experience_level: User experience ("beginner", "intermediate", "advanced", "expert")
            performance_priority: Priority ("stability", "performance", "sample_efficiency")
            
        Returns:
            List[str]: Recommended algorithm names
        """
        recommendations = {
            "beginner": {
                "cooperative": ["vdn", "ippo"],
                "competitive": ["nashq", "minimaxq"],
                "mixed": ["iql", "ippo"]
            },
            "intermediate": {
                "cooperative": ["qmix", "mappo"],
                "competitive": ["wolfphc", "nfsp"],
                "mixed": ["maddpg", "coma"]
            },
            "advanced": {
                "cooperative": ["qtran", "maven"],
                "competitive": ["hql", "lql"],
                "mixed": ["maacc", "dcg"]
            },
            "expert": {
                "cooperative": ["comacomm", "maddpgcomm"],
                "competitive": ["mfq"],
                "mixed": ["comacomm", "maddpgcomm"]
            }
        }
        
        return recommendations.get(experience_level, {}).get(environment_type, ["ippo"])
    
    def get_taxonomy_tree(self) -> Dict:
        """
        Get the complete taxonomy as a tree structure.
        
        Returns:
            Dict: Complete taxonomy tree for visualization
        """
        return self._algorithm_registry
    
    def print_taxonomy(self):
        """Print a formatted view of the complete taxonomy."""
        print("🎯 EasyMARL Algorithm Taxonomy")
        print("=" * 50)
        self._print_tree(self._algorithm_registry, 0)
    
    def _print_tree(self, data, indent_level: int):
        """Recursively print taxonomy tree with proper indentation."""
        indent = "  " * indent_level
        
        if isinstance(data, list):
            for algorithm in data:
                print(f"{indent}📦 {algorithm}")
        elif isinstance(data, dict):
            for key, value in data.items():
                category_name = key.value if hasattr(key, 'value') else str(key)
                print(f"{indent}📁 {category_name}")
                self._print_tree(value, indent_level + 1)


# Global taxonomy instance
taxonomy = AlgorithmTaxonomy()


def get_algorithm_by_taxonomy(category_path: List[str]) -> List[str]:
    """
    Get algorithms by following a specific taxonomy path.
    
    Args:
        category_path: List of category names forming a path
        
    Returns:
        List[str]: Algorithms at the specified path
    """
    return taxonomy.get_algorithms_by_category(*category_path)


def recommend_algorithms(**kwargs) -> List[str]:
    """
    Get algorithm recommendations based on requirements.
    
    Keyword Args:
        environment_type: Type of environment
        experience_level: User experience level
        performance_priority: Performance priority
        
    Returns:
        List[str]: Recommended algorithms
    """
    return taxonomy.get_algorithm_recommendations(**kwargs)


def get_learning_path() -> List[Dict[str, any]]:
    """Get educational learning progression through algorithms."""
    return taxonomy.get_learning_progression()


if __name__ == "__main__":
    # Display the complete taxonomy
    taxonomy.print_taxonomy()
    
    # Show learning progression
    print("\n🎓 Recommended Learning Progression:")
    print("=" * 50)
    for step in taxonomy.get_learning_progression():
        print(f"Level: {step['level']}")
        print(f"Focus: {step['category']}")
        print(f"Algorithms: {', '.join(step['algorithms'])}")
        print(f"Description: {step['description']}")
        print(f"Key Concepts: {', '.join(step['concepts'])}")
        print("-" * 30)
