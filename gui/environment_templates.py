"""
Environment Templates and Utilities for EasyMARL Environment Builder
Provides pre-built environment templates and helper functions for creating custom MultiGrid environments.
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import yaml
import json

@dataclass
class EnvironmentTemplate:
    """Template for creating environments"""
    name: str
    description: str
    category: str
    width: int
    height: int
    n_agents: int
    max_steps: int
    objects: List[Dict]
    agents: List[Dict]
    learning_objectives: List[str]
    difficulty: str
    recommended_algorithms: List[str]

class EnvironmentTemplates:
    """Collection of pre-built environment templates"""
    
    @staticmethod
    def get_all_templates() -> Dict[str, List[EnvironmentTemplate]]:
        """Returns all available templates organized by category"""
        
        templates = {
            'Basic Navigation': [
                EnvironmentTemplates.empty_grid_small(),
                EnvironmentTemplates.empty_grid_medium(),
                EnvironmentTemplates.empty_grid_large(),
                EnvironmentTemplates.simple_maze(),
                EnvironmentTemplates.open_field()
            ],
            'Cooperative Tasks': [
                EnvironmentTemplates.door_key_simple(),
                EnvironmentTemplates.door_key_complex(),
                EnvironmentTemplates.collect_deliver(),
                EnvironmentTemplates.team_navigation(),
                EnvironmentTemplates.four_rooms()
            ],
            'Competitive Games': [
                EnvironmentTemplates.coin_collection(),
                EnvironmentTemplates.tag_game(),
                EnvironmentTemplates.territory_control(),
                EnvironmentTemplates.resource_competition()
            ],
            'Custom Challenges': [
                EnvironmentTemplates.obstacle_course(),
                EnvironmentTemplates.resource_management(),
                EnvironmentTemplates.escape_room(),
                EnvironmentTemplates.stag_hunt(),
                EnvironmentTemplates.coordination_game()
            ]
        }
        
        return templates
    
    @staticmethod
    def empty_grid_small() -> EnvironmentTemplate:
        """6x6 empty grid for basic coordination"""
        return EnvironmentTemplate(
            name="Empty Grid (6x6)",
            description="Simple 6x6 grid for learning basic navigation and coordination. No obstacles, pure multi-agent interaction.",
            category="Basic Navigation",
            width=6, height=6, n_agents=2, max_steps=100,
            objects=[],
            agents=[
                {'x': 1, 'y': 1, 'direction': 0},
                {'x': 4, 'y': 4, 'direction': 2}
            ],
            learning_objectives=[
                "Basic multi-agent coordination",
                "Simple navigation strategies",
                "Collision avoidance",
                "Shared space exploration"
            ],
            difficulty="Beginner",
            recommended_algorithms=["IPPO", "MAPPO", "IQL"]
        )
    
    @staticmethod
    def empty_grid_medium() -> EnvironmentTemplate:
        """8x8 empty grid with more exploration space"""
        return EnvironmentTemplate(
            name="Empty Grid (8x8)",
            description="Medium 8x8 grid providing more exploration space for intermediate coordination learning.",
            category="Basic Navigation",
            width=8, height=8, n_agents=3, max_steps=150,
            objects=[],
            agents=[
                {'x': 1, 'y': 1, 'direction': 0},
                {'x': 6, 'y': 1, 'direction': 1},
                {'x': 1, 'y': 6, 'direction': 3}
            ],
            learning_objectives=[
                "Multi-agent exploration",
                "Coordination with more agents",
                "Spatial awareness",
                "Efficient space coverage"
            ],
            difficulty="Beginner",
            recommended_algorithms=["IPPO", "MAPPO", "VDN"]
        )
    
    @staticmethod
    def door_key_simple() -> EnvironmentTemplate:
        """Simple cooperative door-key scenario"""
        return EnvironmentTemplate(
            name="Door & Key (Simple)",
            description="Agents must cooperate to find keys and unlock doors. One agent finds key, another opens door.",
            category="Cooperative Tasks",
            width=8, height=8, n_agents=2, max_steps=200,
            objects=[
                {'type': 'wall', 'x': 4, 'y': 2, 'color': 'grey'},
                {'type': 'wall', 'x': 4, 'y': 3, 'color': 'grey'},
                {'type': 'wall', 'x': 4, 'y': 4, 'color': 'grey'},
                {'type': 'wall', 'x': 4, 'y': 5, 'color': 'grey'},
                {'type': 'door', 'x': 4, 'y': 3, 'color': 'red', 'locked': True},
                {'type': 'key', 'x': 2, 'y': 2, 'color': 'red'},
                {'type': 'goal', 'x': 6, 'y': 6, 'color': 'green'}
            ],
            agents=[
                {'x': 1, 'y': 1, 'direction': 0},
                {'x': 1, 'y': 6, 'direction': 0}
            ],
            learning_objectives=[
                "Cooperative problem solving",
                "Sequential task coordination",
                "Communication through actions",
                "Goal-oriented behavior"
            ],
            difficulty="Intermediate",
            recommended_algorithms=["QMIX", "VDN", "MAPPO", "MADDPG"]
        )
    
    @staticmethod
    def simple_maze() -> EnvironmentTemplate:
        """Maze navigation requiring exploration"""
        return EnvironmentTemplate(
            name="Simple Maze",
            description="Navigate through a maze with multiple paths. Requires exploration and pathfinding skills.",
            category="Basic Navigation",
            width=10, height=10, n_agents=2, max_steps=250,
            objects=[
                # Outer walls
                {'type': 'wall', 'x': 2, 'y': 2, 'color': 'grey'},
                {'type': 'wall', 'x': 2, 'y': 3, 'color': 'grey'},
                {'type': 'wall', 'x': 2, 'y': 4, 'color': 'grey'},
                {'type': 'wall', 'x': 3, 'y': 4, 'color': 'grey'},
                {'type': 'wall', 'x': 4, 'y': 4, 'color': 'grey'},
                {'type': 'wall', 'x': 4, 'y': 3, 'color': 'grey'},
                {'type': 'wall', 'x': 4, 'y': 2, 'color': 'grey'},
                {'type': 'wall', 'x': 5, 'y': 2, 'color': 'grey'},
                {'type': 'wall', 'x': 6, 'y': 2, 'color': 'grey'},
                {'type': 'wall', 'x': 6, 'y': 3, 'color': 'grey'},
                {'type': 'wall', 'x': 6, 'y': 4, 'color': 'grey'},
                {'type': 'wall', 'x': 6, 'y': 5, 'color': 'grey'},
                {'type': 'wall', 'x': 6, 'y': 6, 'color': 'grey'},
                {'type': 'wall', 'x': 7, 'y': 6, 'color': 'grey'},
                {'type': 'wall', 'x': 8, 'y': 6, 'color': 'grey'},
                {'type': 'goal', 'x': 8, 'y': 8, 'color': 'green'}
            ],
            agents=[
                {'x': 1, 'y': 1, 'direction': 0},
                {'x': 1, 'y': 8, 'direction': 0}
            ],
            learning_objectives=[
                "Maze navigation skills",
                "Exploration strategies",
                "Deadlock avoidance",
                "Pathfinding cooperation"
            ],
            difficulty="Intermediate",
            recommended_algorithms=["IPPO", "QMIX", "DQN"]
        )
    
    @staticmethod
    def coin_collection() -> EnvironmentTemplate:
        """Competitive coin collection game"""
        return EnvironmentTemplate(
            name="Coin Collection",
            description="Agents compete to collect coins scattered across the grid. Competitive multi-agent environment.",
            category="Competitive Games",
            width=10, height=10, n_agents=3, max_steps=200,
            objects=[
                # Scattered coins
                {'type': 'ball', 'x': 3, 'y': 3, 'color': 'yellow'},
                {'type': 'ball', 'x': 7, 'y': 3, 'color': 'yellow'},
                {'type': 'ball', 'x': 5, 'y': 5, 'color': 'yellow'},
                {'type': 'ball', 'x': 2, 'y': 7, 'color': 'yellow'},
                {'type': 'ball', 'x': 8, 'y': 7, 'color': 'yellow'},
                {'type': 'ball', 'x': 4, 'y': 8, 'color': 'yellow'},
                {'type': 'ball', 'x': 6, 'y': 2, 'color': 'yellow'},
                # Some obstacles
                {'type': 'wall', 'x': 5, 'y': 3, 'color': 'grey'},
                {'type': 'wall', 'x': 5, 'y': 4, 'color': 'grey'},
                {'type': 'wall', 'x': 5, 'y': 6, 'color': 'grey'},
                {'type': 'wall', 'x': 5, 'y': 7, 'color': 'grey'}
            ],
            agents=[
                {'x': 1, 'y': 1, 'direction': 0},
                {'x': 8, 'y': 1, 'direction': 2},
                {'x': 1, 'y': 8, 'direction': 1}
            ],
            learning_objectives=[
                "Competitive multi-agent behavior",
                "Resource competition strategies",
                "Efficient collection paths",
                "Opponent modeling"
            ],
            difficulty="Advanced",
            recommended_algorithms=["MADDPG", "NFSP", "PSRO", "MAPPO"]
        )
    
    @staticmethod
    def obstacle_course() -> EnvironmentTemplate:
        """Complex obstacle course for advanced navigation"""
        return EnvironmentTemplate(
            name="Obstacle Course",
            description="Navigate through a complex obstacle course with hazards, moving parts, and multiple challenges.",
            category="Custom Challenges",
            width=12, height=12, n_agents=2, max_steps=300,
            objects=[
                # Starting area walls
                {'type': 'wall', 'x': 3, 'y': 1, 'color': 'grey'},
                {'type': 'wall', 'x': 3, 'y': 2, 'color': 'grey'},
                {'type': 'wall', 'x': 3, 'y': 3, 'color': 'grey'},
                # Narrow passage
                {'type': 'wall', 'x': 6, 'y': 2, 'color': 'grey'},
                {'type': 'wall', 'x': 6, 'y': 4, 'color': 'grey'},
                {'type': 'wall', 'x': 6, 'y': 5, 'color': 'grey'},
                {'type': 'wall', 'x': 6, 'y': 6, 'color': 'grey'},
                # Hazards
                {'type': 'lava', 'x': 4, 'y': 5, 'color': 'red'},
                {'type': 'lava', 'x': 5, 'y': 5, 'color': 'red'},
                {'type': 'lava', 'x': 7, 'y': 8, 'color': 'red'},
                {'type': 'lava', 'x': 8, 'y': 8, 'color': 'red'},
                # Moving boxes (represented as boxes)
                {'type': 'box', 'x': 9, 'y': 4, 'color': 'brown'},
                {'type': 'box', 'x': 9, 'y': 6, 'color': 'brown'},
                # Goal area
                {'type': 'goal', 'x': 10, 'y': 10, 'color': 'green'}
            ],
            agents=[
                {'x': 1, 'y': 1, 'direction': 0},
                {'x': 1, 'y': 2, 'direction': 0}
            ],
            learning_objectives=[
                "Advanced navigation skills",
                "Hazard avoidance",
                "Complex pathfinding",
                "Multi-stage problem solving"
            ],
            difficulty="Expert",
            recommended_algorithms=["MAPPO", "QMIX", "MADDPG"]
        )
    
    @staticmethod
    def four_rooms() -> EnvironmentTemplate:
        """Classic four-room environment"""
        return EnvironmentTemplate(
            name="Four Rooms",
            description="Classic four-room environment requiring coordination to navigate between rooms through doorways.",
            category="Cooperative Tasks",
            width=11, height=11, n_agents=2, max_steps=300,
            objects=[
                # Horizontal wall
                {'type': 'wall', 'x': i, 'y': 5, 'color': 'grey'} for i in range(1, 10) if i != 2 and i != 8
            ] + [
                # Vertical wall
                {'type': 'wall', 'x': 5, 'y': i, 'color': 'grey'} for i in range(1, 10) if i != 3 and i != 7
            ] + [
                # Goals in different rooms
                {'type': 'goal', 'x': 2, 'y': 2, 'color': 'green'},
                {'type': 'goal', 'x': 8, 'y': 8, 'color': 'blue'}
            ],
            agents=[
                {'x': 1, 'y': 1, 'direction': 0},
                {'x': 9, 'y': 9, 'direction': 2}
            ],
            learning_objectives=[
                "Room-to-room navigation",
                "Hierarchical exploration",
                "Long-term planning",
                "Spatial memory"
            ],
            difficulty="Intermediate",
            recommended_algorithms=["QMIX", "VDN", "MAPPO"]
        )
    
    @staticmethod
    def tag_game() -> EnvironmentTemplate:
        """Tag game with pursuit and evasion"""
        return EnvironmentTemplate(
            name="Tag Game",
            description="One agent tries to tag the others in a pursuit-evasion game. Dynamic roles and strategy.",
            category="Competitive Games",
            width=8, height=8, n_agents=3, max_steps=150,
            objects=[
                # Some obstacles for cover
                {'type': 'wall', 'x': 3, 'y': 3, 'color': 'grey'},
                {'type': 'wall', 'x': 4, 'y': 3, 'color': 'grey'},
                {'type': 'wall', 'x': 5, 'y': 3, 'color': 'grey'},
                {'type': 'wall', 'x': 3, 'y': 5, 'color': 'grey'},
                {'type': 'wall', 'x': 4, 'y': 5, 'color': 'grey'},
                {'type': 'wall', 'x': 5, 'y': 5, 'color': 'grey'}
            ],
            agents=[
                {'x': 1, 'y': 1, 'direction': 0},  # Tagger
                {'x': 6, 'y': 1, 'direction': 2},  # Runner 1
                {'x': 1, 'y': 6, 'direction': 1}   # Runner 2
            ],
            learning_objectives=[
                "Pursuit and evasion strategies",
                "Dynamic role adaptation",
                "Predictive behavior modeling",
                "Real-time strategy adjustment"
            ],
            difficulty="Advanced",
            recommended_algorithms=["MADDPG", "NFSP", "PSRO"]
        )
    
    @staticmethod
    def stag_hunt() -> EnvironmentTemplate:
        """Stag hunt coordination game"""
        return EnvironmentTemplate(
            name="Stag Hunt",
            description="Coordination game inspired by the stag hunt dilemma. Agents must cooperate to catch the stag.",
            category="Custom Challenges",
            width=10, height=10, n_agents=3, max_steps=200,
            objects=[
                # Forest environment
                {'type': 'wall', 'x': 3, 'y': 3, 'color': 'green'},
                {'type': 'wall', 'x': 4, 'y': 3, 'color': 'green'},
                {'type': 'wall', 'x': 6, 'y': 3, 'color': 'green'},
                {'type': 'wall', 'x': 7, 'y': 3, 'color': 'green'},
                {'type': 'wall', 'x': 3, 'y': 6, 'color': 'green'},
                {'type': 'wall', 'x': 4, 'y': 6, 'color': 'green'},
                {'type': 'wall', 'x': 6, 'y': 6, 'color': 'green'},
                {'type': 'wall', 'x': 7, 'y': 6, 'color': 'green'},
                # Stag (high value target)
                {'type': 'ball', 'x': 5, 'y': 5, 'color': 'purple'},
                # Hares (low value, individual targets)
                {'type': 'ball', 'x': 2, 'y': 2, 'color': 'brown'},
                {'type': 'ball', 'x': 8, 'y': 2, 'color': 'brown'},
                {'type': 'ball', 'x': 2, 'y': 8, 'color': 'brown'},
                {'type': 'ball', 'x': 8, 'y': 8, 'color': 'brown'}
            ],
            agents=[
                {'x': 1, 'y': 1, 'direction': 0},
                {'x': 8, 'y': 1, 'direction': 2},
                {'x': 1, 'y': 8, 'direction': 1}
            ],
            learning_objectives=[
                "Coordination vs. individual benefit",
                "Trust and cooperation emergence",
                "Risk assessment in cooperation",
                "Social dilemma resolution"
            ],
            difficulty="Expert",
            recommended_algorithms=["MADDPG", "QMIX", "MAPPO", "COMA"]
        )
    
    @staticmethod
    def collect_deliver() -> EnvironmentTemplate:
        """Collect and deliver task"""
        return EnvironmentTemplate(
            name="Collect & Deliver",
            description="Agents must collect items from one area and deliver them to goal zones. Requires coordination and planning.",
            category="Cooperative Tasks",
            width=12, height=8, n_agents=3, max_steps=250,
            objects=[
                # Collection area (left side)
                {'type': 'ball', 'x': 2, 'y': 2, 'color': 'blue'},
                {'type': 'ball', 'x': 2, 'y': 3, 'color': 'red'},
                {'type': 'ball', 'x': 2, 'y': 4, 'color': 'green'},
                {'type': 'ball', 'x': 2, 'y': 5, 'color': 'yellow'},
                {'type': 'ball', 'x': 3, 'y': 2, 'color': 'purple'},
                {'type': 'ball', 'x': 3, 'y': 5, 'color': 'orange'},
                # Barrier
                {'type': 'wall', 'x': 6, 'y': 1, 'color': 'grey'},
                {'type': 'wall', 'x': 6, 'y': 2, 'color': 'grey'},
                {'type': 'wall', 'x': 6, 'y': 4, 'color': 'grey'},
                {'type': 'wall', 'x': 6, 'y': 5, 'color': 'grey'},
                {'type': 'wall', 'x': 6, 'y': 6, 'color': 'grey'},
                # Delivery area (right side)
                {'type': 'goal', 'x': 9, 'y': 2, 'color': 'green'},
                {'type': 'goal', 'x': 9, 'y': 4, 'color': 'green'},
                {'type': 'goal', 'x': 10, 'y': 3, 'color': 'green'}
            ],
            agents=[
                {'x': 1, 'y': 1, 'direction': 0},
                {'x': 1, 'y': 4, 'direction': 0},
                {'x': 1, 'y': 6, 'direction': 0}
            ],
            learning_objectives=[
                "Task division and specialization",
                "Efficient resource transport",
                "Multi-stage coordination",
                "Load balancing strategies"
            ],
            difficulty="Advanced",
            recommended_algorithms=["QMIX", "VDN", "MAPPO", "MADDPG"]
        )
    
    @staticmethod
    def open_field() -> EnvironmentTemplate:
        """Large open field for exploration"""
        return EnvironmentTemplate(
            name="Open Field",
            description="Large open space for advanced exploration and coordination experiments. Suitable for testing coverage algorithms.",
            category="Basic Navigation",
            width=16, height=16, n_agents=4, max_steps=400,
            objects=[
                # Scattered goals for coverage tasks
                {'type': 'goal', 'x': 4, 'y': 4, 'color': 'green'},
                {'type': 'goal', 'x': 12, 'y': 4, 'color': 'green'},
                {'type': 'goal', 'x': 4, 'y': 12, 'color': 'green'},
                {'type': 'goal', 'x': 12, 'y': 12, 'color': 'green'},
                {'type': 'goal', 'x': 8, 'y': 8, 'color': 'blue'}
            ],
            agents=[
                {'x': 1, 'y': 1, 'direction': 0},
                {'x': 14, 'y': 1, 'direction': 2},
                {'x': 1, 'y': 14, 'direction': 1},
                {'x': 14, 'y': 14, 'direction': 3}
            ],
            learning_objectives=[
                "Large space exploration",
                "Area coverage strategies",
                "Long-term coordination",
                "Scalable multi-agent behavior"
            ],
            difficulty="Intermediate",
            recommended_algorithms=["MAPPO", "IPPO", "QMIX"]
        )
    
    @staticmethod
    def territory_control() -> EnvironmentTemplate:
        """Territory control competitive game"""
        return EnvironmentTemplate(
            name="Territory Control",
            description="Agents compete to control territory by occupying key positions. Strategic positioning game.",
            category="Competitive Games",
            width=10, height=10, n_agents=4, max_steps=300,
            objects=[
                # Control points
                {'type': 'goal', 'x': 3, 'y': 3, 'color': 'purple'},
                {'type': 'goal', 'x': 7, 'y': 3, 'color': 'purple'},
                {'type': 'goal', 'x': 3, 'y': 7, 'color': 'purple'},
                {'type': 'goal', 'x': 7, 'y': 7, 'color': 'purple'},
                {'type': 'goal', 'x': 5, 'y': 5, 'color': 'gold'},  # Central control point
                # Some barriers
                {'type': 'wall', 'x': 5, 'y': 2, 'color': 'grey'},
                {'type': 'wall', 'x': 5, 'y': 8, 'color': 'grey'},
                {'type': 'wall', 'x': 2, 'y': 5, 'color': 'grey'},
                {'type': 'wall', 'x': 8, 'y': 5, 'color': 'grey'}
            ],
            agents=[
                {'x': 1, 'y': 1, 'direction': 0},  # Team 1
                {'x': 8, 'y': 1, 'direction': 2},  # Team 1
                {'x': 1, 'y': 8, 'direction': 1},  # Team 2
                {'x': 8, 'y': 8, 'direction': 3}   # Team 2
            ],
            learning_objectives=[
                "Strategic positioning",
                "Territory defense and attack",
                "Team coordination",
                "Resource point control"
            ],
            difficulty="Expert",
            recommended_algorithms=["MADDPG", "MAPPO", "QMIX", "PSRO"]
        )
    
    @staticmethod
    def resource_competition() -> EnvironmentTemplate:
        """Resource competition with limited resources"""
        return EnvironmentTemplate(
            name="Resource Competition",
            description="Limited resources that agents must compete for. Tests competitive vs cooperative strategies.",
            category="Competitive Games",
            width=12, height=8, n_agents=4, max_steps=200,
            objects=[
                # Limited high-value resources
                {'type': 'ball', 'x': 6, 'y': 4, 'color': 'gold'},
                {'type': 'ball', 'x': 5, 'y': 3, 'color': 'silver'},
                {'type': 'ball', 'x': 7, 'y': 3, 'color': 'silver'},
                {'type': 'ball', 'x': 5, 'y': 5, 'color': 'silver'},
                {'type': 'ball', 'x': 7, 'y': 5, 'color': 'silver'},
                # Common resources
                {'type': 'ball', 'x': 3, 'y': 2, 'color': 'brown'},
                {'type': 'ball', 'x': 9, 'y': 2, 'color': 'brown'},
                {'type': 'ball', 'x': 3, 'y': 6, 'color': 'brown'},
                {'type': 'ball', 'x': 9, 'y': 6, 'color': 'brown'},
                # Obstacles
                {'type': 'wall', 'x': 6, 'y': 2, 'color': 'grey'},
                {'type': 'wall', 'x': 6, 'y': 6, 'color': 'grey'}
            ],
            agents=[
                {'x': 1, 'y': 1, 'direction': 0},
                {'x': 10, 'y': 1, 'direction': 2},
                {'x': 1, 'y': 6, 'direction': 1},
                {'x': 10, 'y': 6, 'direction': 3}
            ],
            learning_objectives=[
                "Resource allocation strategies",
                "Competition vs cooperation balance",
                "Value-based decision making",
                "Scarcity-driven behavior"
            ],
            difficulty="Advanced",
            recommended_algorithms=["MADDPG", "NFSP", "MAPPO"]
        )
    
    @staticmethod
    def resource_management() -> EnvironmentTemplate:
        """Complex resource management challenge"""
        return EnvironmentTemplate(
            name="Resource Management",
            description="Complex resource management with multiple resource types, processing, and objectives.",
            category="Custom Challenges",
            width=14, height=10, n_agents=3, max_steps=400,
            objects=[
                # Raw resource deposits
                {'type': 'ball', 'x': 2, 'y': 2, 'color': 'brown'},  # Raw material 1
                {'type': 'ball', 'x': 2, 'y': 3, 'color': 'brown'},
                {'type': 'ball', 'x': 12, 'y': 2, 'color': 'grey'},  # Raw material 2
                {'type': 'ball', 'x': 12, 'y': 3, 'color': 'grey'},
                {'type': 'ball', 'x': 7, 'y': 8, 'color': 'blue'},   # Water/energy
                {'type': 'ball', 'x': 8, 'y': 8, 'color': 'blue'},
                # Processing stations
                {'type': 'box', 'x': 5, 'y': 3, 'color': 'yellow'},  # Processor 1
                {'type': 'box', 'x': 9, 'y': 3, 'color': 'yellow'},  # Processor 2
                {'type': 'box', 'x': 7, 'y': 6, 'color': 'orange'}, # Advanced processor
                # Delivery points
                {'type': 'goal', 'x': 5, 'y': 1, 'color': 'green'},
                {'type': 'goal', 'x': 9, 'y': 1, 'color': 'green'},
                {'type': 'goal', 'x': 7, 'y': 9, 'color': 'purple'}, # Final delivery
                # Transport barriers
                {'type': 'wall', 'x': 4, 'y': 5, 'color': 'grey'},
                {'type': 'wall', 'x': 10, 'y': 5, 'color': 'grey'}
            ],
            agents=[
                {'x': 1, 'y': 5, 'direction': 0},  # Resource collector
                {'x': 7, 'y': 1, 'direction': 1},  # Processor operator
                {'x': 13, 'y': 5, 'direction': 2}  # Transport specialist
            ],
            learning_objectives=[
                "Complex supply chain management",
                "Multi-stage production processes",
                "Resource optimization",
                "Role specialization and coordination"
            ],
            difficulty="Expert",
            recommended_algorithms=["QMIX", "MADDPG", "MAPPO"]
        )
    
    @staticmethod
    def escape_room() -> EnvironmentTemplate:
        """Escape room puzzle challenge"""
        return EnvironmentTemplate(
            name="Escape Room",
            description="Puzzle-solving escape room where agents must work together to find keys, solve puzzles, and escape.",
            category="Custom Challenges",
            width=12, height=12, n_agents=3, max_steps=350,
            objects=[
                # Room boundaries
                {'type': 'wall', 'x': 3, 'y': 3, 'color': 'grey'},
                {'type': 'wall', 'x': 4, 'y': 3, 'color': 'grey'},
                {'type': 'wall', 'x': 5, 'y': 3, 'color': 'grey'},
                {'type': 'wall', 'x': 6, 'y': 3, 'color': 'grey'},
                {'type': 'wall', 'x': 7, 'y': 3, 'color': 'grey'},
                {'type': 'wall', 'x': 8, 'y': 3, 'color': 'grey'},
                {'type': 'wall', 'x': 8, 'y': 4, 'color': 'grey'},
                {'type': 'wall', 'x': 8, 'y': 5, 'color': 'grey'},
                {'type': 'wall', 'x': 8, 'y': 6, 'color': 'grey'},
                {'type': 'wall', 'x': 8, 'y': 7, 'color': 'grey'},
                {'type': 'wall', 'x': 8, 'y': 8, 'color': 'grey'},
                {'type': 'wall', 'x': 7, 'y': 8, 'color': 'grey'},
                {'type': 'wall', 'x': 6, 'y': 8, 'color': 'grey'},
                {'type': 'wall', 'x': 5, 'y': 8, 'color': 'grey'},
                {'type': 'wall', 'x': 4, 'y': 8, 'color': 'grey'},
                {'type': 'wall', 'x': 3, 'y': 8, 'color': 'grey'},
                {'type': 'wall', 'x': 3, 'y': 7, 'color': 'grey'},
                {'type': 'wall', 'x': 3, 'y': 6, 'color': 'grey'},
                {'type': 'wall', 'x': 3, 'y': 5, 'color': 'grey'},
                {'type': 'wall', 'x': 3, 'y': 4, 'color': 'grey'},
                # Puzzle elements
                {'type': 'door', 'x': 7, 'y': 3, 'color': 'red', 'locked': True},   # Exit door
                {'type': 'door', 'x': 5, 'y': 6, 'color': 'blue', 'locked': True}, # Inner door
                {'type': 'key', 'x': 4, 'y': 7, 'color': 'blue'},  # Key for inner door
                {'type': 'key', 'x': 6, 'y': 4, 'color': 'red'},   # Key for exit (behind inner door)
                # Puzzle items
                {'type': 'box', 'x': 4, 'y': 4, 'color': 'purple'}, # Moveable box
                {'type': 'box', 'x': 7, 'y': 7, 'color': 'purple'}, # Moveable box
                # Final goal outside
                {'type': 'goal', 'x': 10, 'y': 2, 'color': 'green'}
            ],
            agents=[
                {'x': 4, 'y': 5, 'direction': 0},
                {'x': 6, 'y': 5, 'direction': 2},
                {'x': 5, 'y': 7, 'direction': 1}
            ],
            learning_objectives=[
                "Sequential puzzle solving",
                "Complex multi-step planning",
                "Cooperative problem decomposition",
                "Spatial reasoning and manipulation"
            ],
            difficulty="Expert",
            recommended_algorithms=["QMIX", "MADDPG", "MAPPO", "COMA"]
        )
    
    @staticmethod
    def coordination_game() -> EnvironmentTemplate:
        """Abstract coordination game"""
        return EnvironmentTemplate(
            name="Coordination Game",
            description="Abstract coordination challenge where agents must synchronize actions to achieve common goals.",
            category="Custom Challenges",
            width=8, height=8, n_agents=4, max_steps=150,
            objects=[
                # Coordination points that require multiple agents
                {'type': 'goal', 'x': 2, 'y': 2, 'color': 'blue'},   # Requires 2 agents
                {'type': 'goal', 'x': 6, 'y': 2, 'color': 'green'},  # Requires 2 agents
                {'type': 'goal', 'x': 2, 'y': 6, 'color': 'yellow'}, # Requires 3 agents
                {'type': 'goal', 'x': 6, 'y': 6, 'color': 'red'},    # Requires all 4 agents
                # Coordination barriers
                {'type': 'wall', 'x': 4, 'y': 1, 'color': 'grey'},
                {'type': 'wall', 'x': 4, 'y': 2, 'color': 'grey'},
                {'type': 'wall', 'x': 4, 'y': 3, 'color': 'grey'},
                {'type': 'wall', 'x': 1, 'y': 4, 'color': 'grey'},
                {'type': 'wall', 'x': 2, 'y': 4, 'color': 'grey'},
                {'type': 'wall', 'x': 3, 'y': 4, 'color': 'grey'},
                {'type': 'wall', 'x': 5, 'y': 4, 'color': 'grey'},
                {'type': 'wall', 'x': 6, 'y': 4, 'color': 'grey'},
                {'type': 'wall', 'x': 7, 'y': 4, 'color': 'grey'},
                {'type': 'wall', 'x': 4, 'y': 5, 'color': 'grey'},
                {'type': 'wall', 'x': 4, 'y': 6, 'color': 'grey'},
                {'type': 'wall', 'x': 4, 'y': 7, 'color': 'grey'}
            ],
            agents=[
                {'x': 1, 'y': 1, 'direction': 0},
                {'x': 7, 'y': 1, 'direction': 2},
                {'x': 1, 'y': 7, 'direction': 1},
                {'x': 7, 'y': 7, 'direction': 3}
            ],
            learning_objectives=[
                "Action synchronization",
                "Implicit coordination protocols",
                "Scalable coordination mechanisms",
                "Emergence of coordination strategies"
            ],
            difficulty="Expert",
            recommended_algorithms=["QMIX", "VDN", "COMA", "MAPPO"]
        )

class EnvironmentExporter:
    """Utilities for exporting environments in different formats"""
    
    @staticmethod
    def to_yaml(template: EnvironmentTemplate, include_training: bool = True) -> str:
        """Export environment template as YAML configuration"""
        
        config = {
            'environment': {
                'name': template.name,
                'type': 'custom_multigrid',
                'description': template.description,
                'category': template.category,
                'width': template.width,
                'height': template.height,
                'n_agents': template.n_agents,
                'max_steps': template.max_steps,
                'agent_view_size': 7,
                'see_through_walls': True
            },
            'agents': template.agents,
            'objects': template.objects,
            'metadata': {
                'learning_objectives': template.learning_objectives,
                'difficulty': template.difficulty,
                'recommended_algorithms': template.recommended_algorithms
            }
        }
        
        if include_training:
            config['training'] = {
                'algorithm': template.recommended_algorithms[0],
                'total_timesteps': 1000000,
                'learning_rate': 0.0003,
                'batch_size': 256,
                'network_type': 'convolutional'
            }
        
        return yaml.dump(config, default_flow_style=False, sort_keys=False)
    
    @staticmethod
    def to_json(template: EnvironmentTemplate) -> str:
        """Export environment template as JSON"""
        
        data = {
            'name': template.name,
            'description': template.description,
            'category': template.category,
            'grid_config': {
                'width': template.width,
                'height': template.height,
                'n_agents': template.n_agents,
                'max_steps': template.max_steps
            },
            'agents': template.agents,
            'objects': template.objects,
            'learning_objectives': template.learning_objectives,
            'difficulty': template.difficulty,
            'recommended_algorithms': template.recommended_algorithms
        }
        
        return json.dumps(data, indent=2)
    
    @staticmethod
    def to_python_class(template: EnvironmentTemplate) -> str:
        """Generate Python class code for the environment"""
        
        class_name = template.name.replace(' ', '').replace('(', '').replace(')', '').replace('&', 'And')
        
        code = f'''"""
Custom MultiGrid Environment: {template.name}
Category: {template.category}
Difficulty: {template.difficulty}

{template.description}

Learning Objectives:
{chr(10).join(f"- {obj}" for obj in template.learning_objectives)}

Recommended Algorithms: {", ".join(template.recommended_algorithms)}
"""

import gym
from gym_multigrid import *

class {class_name}(MultiGridEnv):
    """
    {template.description}
    """
    
    def __init__(self, size={template.width}, n_agents={template.n_agents}, 
                 max_steps={template.max_steps}, **kwargs):
        super().__init__(
            grid_size=size,
            width={template.width},
            height={template.height},
            n_agents=n_agents,
            max_steps=max_steps,
            agent_view_size=7,
            see_through_walls=True,
            **kwargs
        )
    
    def _gen_grid(self, width, height):
        """Generate the environment grid"""
        # Create empty grid
        self.grid = Grid(width, height)
        
        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)
        
        # Place custom objects
'''
        
        # Add object placement code
        for obj in template.objects:
            if obj['type'] == 'wall':
                code += f'        self.grid.set({obj["x"]}, {obj["y"]}, Wall(color="{obj["color"]}"))\n'
            elif obj['type'] == 'door':
                locked = obj.get('locked', False)
                code += f'        self.grid.set({obj["x"]}, {obj["y"]}, Door(color="{obj["color"]}", is_locked={locked}))\n'
            elif obj['type'] == 'key':
                code += f'        self.grid.set({obj["x"]}, {obj["y"]}, Key(color="{obj["color"]}"))\n'
            elif obj['type'] == 'goal':
                code += f'        self.grid.set({obj["x"]}, {obj["y"]}, Goal())\n'
            elif obj['type'] == 'ball':
                code += f'        self.grid.set({obj["x"]}, {obj["y"]}, Ball(color="{obj["color"]}"))\n'
            elif obj['type'] == 'box':
                code += f'        self.grid.set({obj["x"]}, {obj["y"]}, Box(color="{obj["color"]}"))\n'
            elif obj['type'] == 'lava':
                code += f'        self.grid.set({obj["x"]}, {obj["y"]}, Lava())\n'
        
        # Add agent placement code
        code += '\n        # Place agents\n'
        for i, agent in enumerate(template.agents):
            code += f'        self.place_agent({i}, {agent["x"]}, {agent["y"]}, {agent["direction"]})\n'
        
        code += f'''
        self.mission = "{template.description}"

# Training example
if __name__ == "__main__":
    from easymarl.controllers import UnifiedMultiAgentController
    
    # Create environment
    env = {class_name}()
    
    # Initialize EasyMARL controller
    controller = UnifiedMultiAgentController()
    
    # Train with recommended algorithm
    controller.train(
        env=env,
        algorithm="{template.recommended_algorithms[0]}",
        total_timesteps=1000000,
        config={{
            "learning_rate": 0.0003,
            "batch_size": 256,
            "network_type": "convolutional"
        }}
    )
    
    # Evaluate the trained agents
    controller.evaluate(num_episodes=10)
'''
        
        return code
