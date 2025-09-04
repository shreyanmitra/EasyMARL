"""EasyMARL Environments

This module provides environment management and utilities for the EasyMARL framework.
"""

from .vectorized_env import *

try:
    from .gym_multigrid import *
except ImportError:
    pass  # Optional environment


def get_available_environments():
    """Get list of available environments organized by category."""
    environments = {
        'multigrid_basic': [
            'MultiGrid-Empty-6x6-v0',
            'MultiGrid-Empty-8x8-v0',
            'MultiGrid-Empty-16x16-v0'
        ],
        'multigrid_cluttered': [
            'MultiGrid-Cluttered-Fixed-15x15-v0',
            'MultiGrid-Cluttered-Random-10x10-v0'
        ],
        'multigrid_dynamic': [
            'MultiGrid-Dynamic-Obstacles-6x6-v0',
            'MultiGrid-Dynamic-Obstacles-8x8-v0'
        ],
        'multigrid_competitive': [
            'MultiGrid-Competitive-Red-Blue-6x6-v0'
        ]
    }
    
    return environments


def validate_environment(env_name):
    """Validate that an environment can be created and used."""
    try:
        import gymnasium as gym
        
        # Try to create environment
        env = gym.make(env_name)
        
        # Test basic functionality
        obs, info = env.reset()
        
        if hasattr(env, 'action_space'):
            action = env.action_space.sample()
        else:
            action = [env.action_spaces[i].sample() for i in range(env.num_agents)]
            
        obs, reward, terminated, truncated, info = env.step(action)
        
        env.close()
        return True
        
    except Exception as e:
        return False


def get_environment_info(env_name):
    """Get detailed information about an environment."""
    try:
        import gymnasium as gym
        
        env = gym.make(env_name)
        
        info = {
            'name': env_name,
            'observation_space': str(env.observation_space),
            'action_space': str(env.action_space),
            'max_episode_steps': getattr(env, 'max_episode_steps', 'Unknown'),
            'num_agents': getattr(env, 'num_agents', 1)
        }
        
        # MultiGrid specific info
        if 'MultiGrid' in env_name:
            info.update({
                'grid_size': getattr(env, 'grid_size', 'Unknown'),
                'agent_view_size': getattr(env, 'agent_view_size', 'Unknown')
            })
        
        env.close()
        return info
        
    except Exception as e:
        return {'name': env_name, 'error': str(e)}
