#!/usr/bin/env python3
"""
Environment CLI for EasyMARL

This module provides command-line interface for managing environments,
creating custom environments, and exploring available environments.

Usage:
    easymarl-env --list
    easymarl-env --info MultiGrid-Empty-6x6-v0
    easymarl-env --create custom_env --template empty
"""

import argparse
import sys
import os

# Add the parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_env_parser():
    """Create argument parser for environment command."""
    parser = argparse.ArgumentParser(
        description="EasyMARL Environment CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  easymarl-env --list
  easymarl-env --info MultiGrid-Empty-6x6-v0
  easymarl-env --create my_env --template cluttered
  easymarl-env --validate MultiGrid-Custom-v0
        """
    )
    
    # Environment discovery
    parser.add_argument('--list', action='store_true',
                       help='List all available environments')
    parser.add_argument('--info', type=str, default=None,
                       help='Show detailed information about an environment')
    
    # Environment creation
    parser.add_argument('--create', type=str, default=None,
                       help='Create new environment with given name')
    parser.add_argument('--template', type=str, default='empty',
                       choices=['empty', 'cluttered', 'dynamic', 'competitive'],
                       help='Template for new environment (default: empty)')
    
    # Environment validation
    parser.add_argument('--validate', type=str, default=None,
                       help='Validate environment implementation')
    parser.add_argument('--test', type=str, default=None,
                       help='Test environment with random actions')
    
    # Environment configuration
    parser.add_argument('--size', type=str, default='6x6',
                       help='Environment size for creation (default: 6x6)')
    parser.add_argument('--agents', type=int, default=2,
                       help='Number of agents (default: 2)')
    
    return parser

def list_environments():
    """List all available environments."""
    try:
        import gymnasium as gym
        from environments import get_available_environments
        
        print("🌍 Available EasyMARL Environments:")
        print("=" * 50)
        
        # Get MultiGrid environments
        envs = get_available_environments()
        
        for category, env_list in envs.items():
            print(f"\n{category.upper()}:")
            for env_name in env_list:
                print(f"  • {env_name}")
                
        return 0
    except ImportError as e:
        print(f"❌ Failed to import environment modules: {e}")
        # Fallback to basic list
        basic_envs = [
            "MultiGrid-Empty-6x6-v0",
            "MultiGrid-Empty-8x8-v0", 
            "MultiGrid-Cluttered-Fixed-15x15-v0",
            "MultiGrid-Dynamic-Obstacles-6x6-v0"
        ]
        print("🌍 Basic MultiGrid Environments:")
        for env in basic_envs:
            print(f"  • {env}")
        return 0
    except Exception as e:
        print(f"❌ Failed to list environments: {e}")
        return 1

def show_env_info(env_name):
    """Show detailed information about an environment."""
    try:
        import gymnasium as gym
        
        print(f"🔍 Environment Information: {env_name}")
        print("=" * 50)
        
        # Try to create environment
        env = gym.make(env_name)
        
        print(f"Observation Space: {env.observation_space}")
        print(f"Action Space: {env.action_space}")
        
        if hasattr(env, 'num_agents'):
            print(f"Number of Agents: {env.num_agents}")
        if hasattr(env, 'max_episode_steps'):
            print(f"Max Episode Steps: {env.max_episode_steps}")
            
        # Environment-specific info
        if 'MultiGrid' in env_name:
            if hasattr(env, 'grid_size'):
                print(f"Grid Size: {env.grid_size}")
            if hasattr(env, 'agent_view_size'):
                print(f"Agent View Size: {env.agent_view_size}")
                
        env.close()
        return 0
        
    except Exception as e:
        print(f"❌ Failed to get environment info: {e}")
        return 1

def create_environment(name, template, size, num_agents):
    """Create a new environment."""
    try:
        from gui.environment_builder import create_custom_environment
        
        print(f"🏗️ Creating environment: {name}")
        print(f"   Template: {template}")
        print(f"   Size: {size}")
        print(f"   Agents: {num_agents}")
        
        config = {
            'name': name,
            'template': template,
            'size': size,
            'num_agents': num_agents
        }
        
        env_file = create_custom_environment(config)
        print(f"✅ Environment created: {env_file}")
        return 0
        
    except ImportError:
        print("❌ Environment builder not available")
        print("💡 Use the GUI interface for environment creation")
        return 1
    except Exception as e:
        print(f"❌ Environment creation failed: {e}")
        return 1

def validate_environment(env_name):
    """Validate environment implementation."""
    try:
        import gymnasium as gym
        import numpy as np
        
        print(f"🧪 Validating environment: {env_name}")
        
        env = gym.make(env_name)
        
        # Test reset
        obs, info = env.reset()
        print("✅ Reset successful")
        
        # Test step
        if hasattr(env, 'action_space'):
            action = env.action_space.sample()
        else:
            action = [env.action_spaces[i].sample() for i in range(env.num_agents)]
            
        obs, reward, terminated, truncated, info = env.step(action)
        print("✅ Step successful")
        
        # Test observation space
        if hasattr(env, 'observation_space'):
            assert env.observation_space.contains(obs), "Observation space validation failed"
        print("✅ Observation space valid")
        
        # Test action space
        if hasattr(env, 'action_space'):
            test_action = env.action_space.sample()
            assert env.action_space.contains(test_action), "Action space validation failed"
        print("✅ Action space valid")
        
        env.close()
        print("✅ Environment validation passed")
        return 0
        
    except Exception as e:
        print(f"❌ Environment validation failed: {e}")
        return 1

def test_environment(env_name, episodes=5):
    """Test environment with random actions."""
    try:
        import gymnasium as gym
        import numpy as np
        
        print(f"🎮 Testing environment: {env_name}")
        print(f"Running {episodes} episodes with random actions...")
        
        env = gym.make(env_name)
        
        for episode in range(episodes):
            obs, info = env.reset()
            total_reward = 0
            steps = 0
            
            while True:
                if hasattr(env, 'action_space'):
                    action = env.action_space.sample()
                else:
                    action = [env.action_spaces[i].sample() for i in range(env.num_agents)]
                    
                obs, reward, terminated, truncated, info = env.step(action)
                
                if isinstance(reward, (list, tuple)):
                    total_reward += sum(reward)
                else:
                    total_reward += reward
                    
                steps += 1
                
                if terminated or truncated:
                    break
                    
            print(f"  Episode {episode + 1}: {steps} steps, reward: {total_reward:.2f}")
            
        env.close()
        print("✅ Environment test completed")
        return 0
        
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        return 1

def main():
    """Main environment CLI entry point."""
    parser = create_env_parser()
    args = parser.parse_args()
    
    if args.list:
        return list_environments()
    elif args.info:
        return show_env_info(args.info)
    elif args.create:
        return create_environment(args.create, args.template, args.size, args.agents)
    elif args.validate:
        return validate_environment(args.validate)
    elif args.test:
        return test_environment(args.test)
    else:
        parser.print_help()
        return 0

if __name__ == '__main__':
    sys.exit(main())
