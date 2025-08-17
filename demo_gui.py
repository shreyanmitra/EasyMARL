"""
Demonstration script for the enhanced EasyMARL GUI.

This script shows how to use the new GUI features including:
- Real-time training graphs
- Algorithm descriptions
- WandB integration
- Downloadable training data
"""

import gradio as gr
import yaml
import os
from gui import interface

def create_demo_config():
    """Create a demo configuration for quick testing."""
    demo_config = {
        "qmix_demo": {
            "mode": "QMIX",
            "algorithm": "qmix",
            "max_episodes": 500,
            "max_steps": 50,
            "learning_rate": 0.0005,
            "gamma": 0.99,
            "epsilon_start": 1.0,
            "epsilon_end": 0.05,
            "epsilon_decay": 0.995,
            "batch_size": 32,
            "memory_size": 5000,
            "hidden_dim": 64,
            "use_wandb": True,
            "wandb_project": "easymarl_demo"
        },
        "ippo_demo": {
            "mode": "IPPO", 
            "algorithm": "ippo",
            "max_episodes": 300,
            "max_steps": 50,
            "learning_rate": 0.001,
            "gamma": 0.99,
            "lambda_gae": 0.95,
            "clip_epsilon": 0.2,
            "value_loss_coef": 0.5,
            "entropy_coef": 0.01,
            "ppo_epochs": 4,
            "mini_batch_size": 64,
            "use_wandb": True,
            "wandb_project": "easymarl_demo"
        }
    }
    
    # Save demo configs
    os.makedirs("config/demo", exist_ok=True)
    for name, config in demo_config.items():
        with open(f"config/demo/{name}.yaml", 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    print("Demo configurations created in config/demo/")

def launch_demo():
    """Launch the GUI in demo mode."""
    print("🚀 Launching EasyMARL GUI Demo")
    print("=" * 50)
    
    print("✨ Enhanced Features:")
    print("• 21+ MARL algorithms with detailed descriptions")
    print("• Real-time training graphs and metrics")  
    print("• Environment visualization")
    print("• WandB integration for experiment tracking")
    print("• Downloadable training data")
    print("• Comprehensive algorithm catalog")
    print("• Professional UI with tabbed interface")
    
    print("\n🎯 Quick Start Guide:")
    print("1. Go to 'Training' tab")
    print("2. Select algorithm (try QMIX or IPPO)")
    print("3. Configure episodes and learning rate")
    print("4. Enable WandB logging for detailed tracking")
    print("5. Click 'Start Training' and watch real-time progress!")
    
    print("\n📊 Training Features:")
    print("• Live graphs update every 2 seconds")
    print("• Episode rewards and lengths tracking")
    print("• Environment state visualization")
    print("• Training status and progress indicators")
    print("• Downloadable JSON data for analysis")
    
    print("\n📚 Algorithm Catalog:")
    print("• Detailed descriptions for each algorithm")
    print("• Algorithm types and best use cases")
    print("• Pros/cons for informed selection")
    print("• Feature lists and technical details")
    
    print("\n" + "=" * 50)
    print("🌐 Starting GUI on localhost...")
    
    # Create demo configs
    create_demo_config()
    
    # Launch the interface
    interface.launch(
        share=True,
        debug=False,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )

if __name__ == "__main__":
    launch_demo()
