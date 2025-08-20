"""
GUI Environment Builder for EasyMARL
Creates custom MultiGrid environments through an intuitive interface.
Supports grid design, object placement, and environment configuration.
"""

import gradio as gr
import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import copy

@dataclass
class GridConfig:
    """Configuration for a custom MultiGrid environment"""
    name: str
    width: int
    height: int
    n_agents: int
    max_steps: int
    see_through_walls: bool = True
    agent_view_size: int = 7
    description: str = ""

@dataclass  
class GridObject:
    """Represents an object in the grid"""
    type: str  # 'wall', 'door', 'key', 'ball', 'box', 'goal', 'lava'
    x: int
    y: int
    color: str = 'red'
    locked: bool = False  # For doors
    
@dataclass
class AgentConfig:
    """Configuration for an agent"""
    x: int
    y: int
    direction: int = 0  # 0=right, 1=down, 2=left, 3=up

class EnvironmentBuilder:
    """Main class for building custom MultiGrid environments"""
    
    def __init__(self):
        self.grid_config = GridConfig("Custom-Environment", 8, 8, 2, 100)
        self.grid_objects: List[GridObject] = []
        self.agents: List[AgentConfig] = [
            AgentConfig(1, 1), 
            AgentConfig(6, 6)
        ]
        self.grid_state = np.zeros((8, 8), dtype=int)
        
        # Object type mappings for visualization
        self.object_types = {
            'empty': 0,
            'wall': 1, 
            'door': 2,
            'key': 3,
            'ball': 4,
            'box': 5,
            'goal': 6,
            'lava': 7,
            'agent': 8
        }
        
        # Color mappings
        self.colors = ['red', 'green', 'blue', 'purple', 'yellow', 'grey']
        
    def create_interface(self):
        """Creates the Gradio interface for environment building"""
        
        with gr.Blocks(title="EasyMARL Environment Builder", theme=gr.themes.Soft()) as interface:
            gr.Markdown("# 🏗️ EasyMARL Environment Builder")
            gr.Markdown("Create custom MultiGrid environments with an intuitive drag-and-drop interface.")
            
            with gr.Tab("🎯 Grid Designer"):
                self._create_grid_designer_tab()
                
            with gr.Tab("⚙️ Environment Config"):
                self._create_config_tab()
                
            with gr.Tab("🤖 Agent Setup"):
                self._create_agent_tab()
                
            with gr.Tab("💾 Save & Export"):
                self._create_export_tab()
                
            with gr.Tab("📋 Templates"):
                self._create_templates_tab()
                
        return interface
    
    def _create_grid_designer_tab(self):
        """Creates the main grid design interface"""
        
        with gr.Row():
            # Left panel - Tools
            with gr.Column(scale=1):
                gr.Markdown("### 🛠️ Tools")
                
                selected_tool = gr.Radio(
                    choices=['wall', 'door', 'key', 'ball', 'box', 'goal', 'lava', 'eraser'],
                    value='wall',
                    label="Select Tool"
                )
                
                selected_color = gr.Dropdown(
                    choices=self.colors,
                    value='red',
                    label="Object Color"
                )
                
                gr.Markdown("### 📏 Grid Size")
                grid_width = gr.Slider(
                    minimum=5, maximum=20, value=8, step=1,
                    label="Width"
                )
                grid_height = gr.Slider(
                    minimum=5, maximum=20, value=8, step=1,
                    label="Height"
                )
                
                resize_btn = gr.Button("Resize Grid", variant="secondary")
                clear_btn = gr.Button("Clear Grid", variant="stop")
                
            # Right panel - Grid Canvas
            with gr.Column(scale=3):
                gr.Markdown("### 🎨 Grid Canvas")
                gr.Markdown("Click on cells to place objects. Use the tools on the left to select what to place.")
                
                # Grid visualization as HTML
                grid_html = gr.HTML(
                    value=self._generate_grid_html(),
                    label="Grid Canvas"
                )
                
                # Coordinate display
                coord_display = gr.Textbox(
                    value="Click on grid to see coordinates",
                    label="Current Position",
                    interactive=False
                )
        
        # Event handlers
        resize_btn.click(
            fn=self._resize_grid,
            inputs=[grid_width, grid_height],
            outputs=[grid_html]
        )
        
        clear_btn.click(
            fn=self._clear_grid,
            outputs=[grid_html]
        )
        
    def _create_config_tab(self):
        """Creates environment configuration interface"""
        
        with gr.Row():
            with gr.Column():
                env_name = gr.Textbox(
                    value=self.grid_config.name,
                    label="Environment Name",
                    placeholder="My-Custom-Environment"
                )
                
                env_description = gr.Textbox(
                    value=self.grid_config.description,
                    label="Description",
                    placeholder="Description of your environment...",
                    lines=3
                )
                
                max_steps = gr.Slider(
                    minimum=50, maximum=1000, value=100, step=50,
                    label="Max Steps per Episode"
                )
                
            with gr.Column():
                n_agents = gr.Slider(
                    minimum=2, maximum=8, value=2, step=1,
                    label="Number of Agents"
                )
                
                agent_view_size = gr.Slider(
                    minimum=3, maximum=15, value=7, step=2,
                    label="Agent View Size (must be odd)"
                )
                
                see_through_walls = gr.Checkbox(
                    value=True,
                    label="Agents can see through walls"
                )
                
        # Preview section
        with gr.Row():
            config_preview = gr.JSON(
                value=asdict(self.grid_config),
                label="Current Configuration"
            )
            
    def _create_agent_tab(self):
        """Creates agent configuration interface"""
        
        gr.Markdown("### 🤖 Agent Positions & Orientations")
        gr.Markdown("Configure starting positions and orientations for each agent.")
        
        # Dynamic agent configuration based on n_agents
        agent_configs = []
        for i in range(8):  # Max 8 agents
            with gr.Row(visible=(i < self.grid_config.n_agents)):
                gr.Markdown(f"**Agent {i+1}**")
                x_pos = gr.Slider(1, 7, value=1, step=1, label="X Position")
                y_pos = gr.Slider(1, 7, value=1, step=1, label="Y Position")
                direction = gr.Radio(
                    choices=['→ Right', '↓ Down', '← Left', '↑ Up'],
                    value='→ Right',
                    label="Initial Direction"
                )
                agent_configs.append((x_pos, y_pos, direction))
        
        # Agent placement visualization
        agent_preview = gr.HTML(
            value=self._generate_agent_preview_html(),
            label="Agent Positions Preview"
        )
        
    def _create_export_tab(self):
        """Creates environment export interface"""
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 💾 Export Options")
                
                export_format = gr.Radio(
                    choices=['YAML Config', 'Python Code', 'JSON Data'],
                    value='YAML Config',
                    label="Export Format"
                )
                
                include_training = gr.Checkbox(
                    value=True,
                    label="Include training configuration template"
                )
                
                export_btn = gr.Button("Generate Export", variant="primary")
                
            with gr.Column():
                gr.Markdown("### 📁 File Output")
                
                export_content = gr.Code(
                    value="# Click 'Generate Export' to see the configuration",
                    language="yaml",
                    label="Generated Configuration"
                )
                
                download_btn = gr.DownloadButton(
                    label="Download Configuration",
                    visible=False
                )
        
        # Event handlers
        export_btn.click(
            fn=self._generate_export,
            inputs=[export_format, include_training],
            outputs=[export_content, download_btn]
        )
        
    def _create_templates_tab(self):
        """Creates environment template interface"""
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📋 Environment Templates")
                gr.Markdown("Load pre-built environment templates as starting points.")
                
                template_category = gr.Dropdown(
                    choices=['Basic Navigation', 'Cooperative Tasks', 'Competitive Games', 'Custom Challenges'],
                    value='Basic Navigation',
                    label="Template Category"
                )
                
                template_list = gr.Dropdown(
                    choices=self._get_template_names('Basic Navigation'),
                    label="Available Templates"
                )
                
                load_template_btn = gr.Button("Load Template", variant="secondary")
                
            with gr.Column():
                template_preview = gr.Image(
                    label="Template Preview",
                    type="numpy"
                )
                
                template_description = gr.Markdown(
                    "Select a template to see its description and preview."
                )
        
        # Event handlers
        template_category.change(
            fn=self._get_template_names,
            inputs=[template_category],
            outputs=[template_list]
        )
        
        template_list.change(
            fn=self._preview_template,
            inputs=[template_list],
            outputs=[template_preview, template_description]
        )
        
        load_template_btn.click(
            fn=self._load_template,
            inputs=[template_list],
            outputs=[]  # Will update multiple components
        )
    
    def _generate_grid_html(self) -> str:
        """Generates HTML representation of the grid"""
        
        html = """
        <style>
        .grid-container {
            display: inline-block;
            border: 2px solid #333;
            background: #f0f0f0;
        }
        .grid-row {
            display: flex;
        }
        .grid-cell {
            width: 30px;
            height: 30px;
            border: 1px solid #ccc;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            font-size: 12px;
            font-weight: bold;
        }
        .wall { background-color: #666; }
        .door { background-color: #8B4513; }
        .key { background-color: #FFD700; }
        .ball { background-color: #FF6B6B; border-radius: 50%; }
        .box { background-color: #DDD; }
        .goal { background-color: #4ECDC4; }
        .lava { background-color: #FF4444; }
        .agent { background-color: #4CAF50; border-radius: 50%; }
        .empty { background-color: white; }
        </style>
        
        <div class="grid-container">
        """
        
        for y in range(self.grid_config.height):
            html += '<div class="grid-row">'
            for x in range(self.grid_config.width):
                cell_type = self._get_cell_type(x, y)
                cell_content = self._get_cell_content(x, y)
                
                html += f'<div class="grid-cell {cell_type}" onclick="selectCell({x}, {y})">{cell_content}</div>'
            html += '</div>'
        
        html += '</div>'
        
        # Add JavaScript for cell selection
        html += """
        <script>
        function selectCell(x, y) {
            document.getElementById('coord-display').value = `Selected: (${x}, ${y})`;
            // Here we would trigger the cell placement logic
        }
        </script>
        """
        
        return html
    
    def _get_cell_type(self, x: int, y: int) -> str:
        """Returns the CSS class for a grid cell"""
        
        # Check if there's an agent at this position
        for agent in self.agents:
            if agent.x == x and agent.y == y:
                return 'agent'
        
        # Check for objects
        for obj in self.grid_objects:
            if obj.x == x and obj.y == y:
                return obj.type
        
        return 'empty'
    
    def _get_cell_content(self, x: int, y: int) -> str:
        """Returns the content to display in a grid cell"""
        
        # Check for agents first
        for i, agent in enumerate(self.agents):
            if agent.x == x and agent.y == y:
                directions = ['→', '↓', '←', '↑']
                return f"A{i+1}"
        
        # Check for objects
        for obj in self.grid_objects:
            if obj.x == x and obj.y == y:
                symbols = {
                    'wall': '█',
                    'door': '🚪',
                    'key': '🔑',
                    'ball': '⚽',
                    'box': '📦',
                    'goal': '🎯',
                    'lava': '🔥'
                }
                return symbols.get(obj.type, '?')
        
        return ''
    
    def _resize_grid(self, width: int, height: int) -> str:
        """Resizes the grid and returns new HTML"""
        self.grid_config.width = width
        self.grid_config.height = height
        
        # Remove objects that are now outside the grid
        self.grid_objects = [
            obj for obj in self.grid_objects 
            if obj.x < width and obj.y < height
        ]
        
        # Move agents that are outside the grid
        for agent in self.agents:
            if agent.x >= width:
                agent.x = width - 1
            if agent.y >= height:
                agent.y = height - 1
        
        return self._generate_grid_html()
    
    def _clear_grid(self) -> str:
        """Clears all objects from the grid"""
        self.grid_objects.clear()
        return self._generate_grid_html()
    
    def _generate_agent_preview_html(self) -> str:
        """Generates HTML preview of agent positions"""
        html = "<div style='font-family: monospace;'>"
        for i, agent in enumerate(self.agents[:self.grid_config.n_agents]):
            direction_names = ['Right', 'Down', 'Left', 'Up']
            html += f"<p><strong>Agent {i+1}:</strong> Position ({agent.x}, {agent.y}), Facing {direction_names[agent.direction]}</p>"
        html += "</div>"
        return html
    
    def _get_template_names(self, category: str) -> List[str]:
        """Returns template names for a given category"""
        templates = {
            'Basic Navigation': ['Empty Grid', 'Simple Maze', 'Open Field'],
            'Cooperative Tasks': ['Door & Key', 'Collect & Deliver', 'Team Navigation'],
            'Competitive Games': ['Coin Collection', 'Tag Game', 'Territory Control'],
            'Custom Challenges': ['Obstacle Course', 'Resource Management', 'Escape Room']
        }
        return templates.get(category, [])
    
    def _preview_template(self, template_name: str) -> Tuple[np.ndarray, str]:
        """Generates preview and description for a template"""
        # This would generate actual template previews
        # For now, return placeholder
        preview_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        description = f"**{template_name}**\n\nThis template provides a {template_name.lower()} environment suitable for multi-agent reinforcement learning experiments."
        return preview_image, description
    
    def _load_template(self, template_name: str):
        """Loads a template into the current environment"""
        # This would load actual template configurations
        # Implementation would populate self.grid_config, self.grid_objects, etc.
        pass
    
    def _generate_export(self, export_format: str, include_training: bool) -> Tuple[str, bool]:
        """Generates export content in the specified format"""
        
        if export_format == 'YAML Config':
            content = self._generate_yaml_config(include_training)
            language = 'yaml'
        elif export_format == 'Python Code':
            content = self._generate_python_code(include_training)
            language = 'python'
        else:  # JSON Data
            content = self._generate_json_data()
            language = 'json'
        
        return content, True  # True makes download button visible
    
    def _generate_yaml_config(self, include_training: bool) -> str:
        """Generates YAML configuration for the environment"""
        
        config = {
            'environment': {
                'name': self.grid_config.name,
                'type': 'custom_multigrid',
                'description': self.grid_config.description,
                'width': self.grid_config.width,
                'height': self.grid_config.height,
                'n_agents': self.grid_config.n_agents,
                'max_steps': self.grid_config.max_steps,
                'agent_view_size': self.grid_config.agent_view_size,
                'see_through_walls': self.grid_config.see_through_walls
            },
            'agents': [
                {
                    'id': i,
                    'start_x': agent.x,
                    'start_y': agent.y,
                    'start_direction': agent.direction
                }
                for i, agent in enumerate(self.agents[:self.grid_config.n_agents])
            ],
            'objects': [
                {
                    'type': obj.type,
                    'x': obj.x,
                    'y': obj.y,
                    'color': obj.color,
                    'locked': obj.locked if hasattr(obj, 'locked') else False
                }
                for obj in self.grid_objects
            ]
        }
        
        if include_training:
            config['training'] = {
                'algorithm': 'IPPO',
                'total_timesteps': 1000000,
                'learning_rate': 0.0003,
                'batch_size': 256,
                'network_type': 'convolutional'
            }
        
        # Convert to YAML string
        import yaml
        return yaml.dump(config, default_flow_style=False, sort_keys=False)
    
    def _generate_python_code(self, include_training: bool) -> str:
        """Generates Python code to create the environment"""
        
        code = f'''"""
Custom MultiGrid Environment: {self.grid_config.name}
Generated by EasyMARL Environment Builder
"""

import gym
from gym_multigrid import *

class {self.grid_config.name.replace('-', '')}(MultiGridEnv):
    """
    {self.grid_config.description}
    """
    
    def __init__(self, size={self.grid_config.width}, n_agents={self.grid_config.n_agents}, 
                 max_steps={self.grid_config.max_steps}, **kwargs):
        super().__init__(
            grid_size=size,
            width={self.grid_config.width},
            height={self.grid_config.height},
            n_agents=n_agents,
            max_steps=max_steps,
            agent_view_size={self.grid_config.agent_view_size},
            see_through_walls={str(self.grid_config.see_through_walls)},
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
        for obj in self.grid_objects:
            if obj.type == 'wall':
                code += f'        self.grid.set({obj.x}, {obj.y}, Wall(color="{obj.color}"))\n'
            elif obj.type == 'door':
                code += f'        self.grid.set({obj.x}, {obj.y}, Door(color="{obj.color}", is_locked={obj.locked}))\n'
            elif obj.type == 'key':
                code += f'        self.grid.set({obj.x}, {obj.y}, Key(color="{obj.color}"))\n'
            elif obj.type == 'goal':
                code += f'        self.grid.set({obj.x}, {obj.y}, Goal())\n'
            # Add more object types as needed
        
        # Add agent placement code
        code += '\n        # Place agents\n'
        for i, agent in enumerate(self.agents[:self.grid_config.n_agents]):
            code += f'        self.place_agent({i}, {agent.x}, {agent.y}, {agent.direction})\n'
        
        code += '\n        self.mission = "Navigate and complete the task"\n\n'
        
        if include_training:
            code += '''
# Training script
if __name__ == "__main__":
    from easymarl import EasyMARL
    
    # Create environment
    env_name = "''' + self.grid_config.name + '''"
    
    # Initialize EasyMARL
    marl = EasyMARL(
        env_name=env_name,
        algorithm="IPPO",
        total_timesteps=1000000,
        config={
            "learning_rate": 0.0003,
            "batch_size": 256,
            "network_type": "convolutional"
        }
    )
    
    # Train the agents
    marl.train()
    
    # Evaluate
    marl.evaluate(num_episodes=10)
'''
        
        return code
    
    def _generate_json_data(self) -> str:
        """Generates JSON data representation"""
        
        data = {
            'environment_config': asdict(self.grid_config),
            'agents': [asdict(agent) for agent in self.agents[:self.grid_config.n_agents]],
            'objects': [asdict(obj) for obj in self.grid_objects],
            'metadata': {
                'created_with': 'EasyMARL Environment Builder',
                'version': '1.0.0',
                'compatible_algorithms': [
                    'IPPO', 'MAPPO', 'MADDPG', 'QMIX', 'VDN', 'IQL', 'DQN'
                ]
            }
        }
        
        import json
        return json.dumps(data, indent=2)

# Create the interface
def create_environment_builder():
    """Creates and returns the environment builder interface"""
    builder = EnvironmentBuilder()
    return builder.create_interface()

if __name__ == "__main__":
    interface = create_environment_builder()
    interface.launch(share=False, server_name="0.0.0.0", server_port=7861)
