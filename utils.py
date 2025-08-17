"""
(C) Shreyan Mitra, based on starter code by Natasha Jaques

EasyMARL Utility Functions and Helper Classes

This module provides essential utility functions used throughout the EasyMARL framework.
These utilities handle common tasks like configuration management, environment creation,
data processing, and visualization.

Key Components:
1. Configuration Management: Loading and merging YAML configs
2. Environment Creation: Factory functions for different environment types
3. Data Processing: Tensor operations and state preprocessing
4. Visualization: Plotting and video generation utilities
5. Reproducibility: Seed setting and deterministic operations

For MARL Beginners:
These are the "helper tools" that make the framework easier to use. You don't need
to understand every detail initially, but they handle important background tasks
like setting up environments and processing data.
"""

# Import essential libraries for the framework
import gym                     # OpenAI Gym for reinforcement learning environments
from matplotlib.gridspec import GridSpec  # For creating subplot layouts
from matplotlib import pyplot as plt      # For plotting and visualization
from moviepy.editor import *              # For video creation and editing
import numpy as np            # Numerical computations
import os                     # Operating system interface
import random                 # Random number generation
import seaborn as sns         # Statistical data visualization
import torch                  # PyTorch for deep learning
import wandb                  # Weights & Biases for experiment tracking
import yaml                   # YAML configuration file parsing


class dotdict(dict):
    """
    Dictionary with dot notation access to attributes.
    
    This utility class allows accessing dictionary keys using dot notation,
    making configuration objects more convenient to use.
    
    Example:
        config = dotdict({'learning_rate': 0.001, 'gamma': 0.99})
        print(config.learning_rate)  # Instead of config['learning_rate']
    
    For Beginners:
    This makes configuration objects easier to work with. Instead of writing
    config['learning_rate'], you can write config.learning_rate.
    """
    __getattr__ = dict.get    # Allow config.key syntax for getting values
    __setattr__ = dict.__setitem__  # Allow config.key = value syntax for setting
    __delattr__ = dict.__delitem__  # Allow del config.key syntax for deletion


def merge_configs(update, default):
    """
    Recursively merge two configuration dictionaries.
    
    This function combines a default configuration with updates, ensuring that
    all default values are preserved unless explicitly overridden. It handles
    nested dictionaries properly by recursively merging them.
    
    Args:
        update (dict): Configuration updates/overrides
        default (dict): Default configuration values
    
    Returns:
        dict: Merged configuration with updates applied to defaults
    
    Example:
        default = {'algo': {'lr': 0.001, 'gamma': 0.99}, 'env': 'MultiGrid'}
        update = {'algo': {'lr': 0.01}}
        result = merge_configs(update, default)
        # Result: {'algo': {'lr': 0.01, 'gamma': 0.99}, 'env': 'MultiGrid'}
    
    For Beginners:
    This ensures you get sensible defaults for all settings while still being
    able to customize specific parameters. Like having a template with some
    custom modifications.
    """
    if isinstance(update, dict) and isinstance(default, dict):
        # Both are dictionaries, so merge them recursively
        for k, v in default.items():
            if k not in update:
                # Key not in update, use default value
                update[k] = v
            else:
                # Key exists in both, merge recursively
                update[k] = merge_configs(update[k], v)
    return update


def make_env(config):
    """
    Factory function to create environments based on configuration.
    
    This function creates and returns the appropriate environment based on the
    domain specified in the configuration. It handles different environment
    types and their specific initialization requirements.
    
    Args:
        config: Configuration object containing environment specification
                Must have 'domain' attribute specifying environment name
    
    Returns:
        gym.Env: Initialized environment ready for training
    
    Raises:
        NotImplementedError: If environment type is not supported
    
    Example:
        config = dotdict({'domain': 'MultiGrid-Cluttered-Fixed-15x15'})
        env = make_env(config)
    
    For Beginners:
    This is like a "environment factory" that creates the right type of
    environment for your experiment. Just specify the environment name
    in your config and this function handles the setup.
    """
    if 'MultiGrid' in config.domain:
        # MultiGrid environments (grid-based multi-agent environments)
        from envs import gym_multigrid
        from envs.gym_multigrid import multigrid_envs
        
        # Create environment using OpenAI Gym interface
        env = gym.make(config.domain)
        print(f"Created MultiGrid environment: {config.domain}")
        return env
    else:
        # Environment type not yet supported
        raise NotImplementedError(f"Environment {config.domain} not implemented yet")


def argmax_2d_index(arr):
    """
    Find the 2D index of the maximum value in a 2D tensor.
    
    This function finds the (row, column) coordinates of the maximum value
    in a 2D tensor. If there are multiple maximum values, it randomly
    selects one to break ties.
    
    Args:
        arr (torch.Tensor): 2D tensor to find maximum in
    
    Returns:
        torch.Tensor: 1D tensor containing [row, column] of maximum value
    
    Example:
        arr = torch.tensor([[1, 3], [2, 4]])
        idx = argmax_2d_index(arr)  # Returns [1, 1] (position of value 4)
    
    For Beginners:
    This is useful for finding the best action in 2D action spaces or
    locating the most important position in a grid-like representation.
    """
    assert len(arr.shape) == 2, "Input must be a 2D tensor"
    
    # Find all positions where the value equals the maximum
    best_2d_index = (arr == torch.max(arr)).nonzero()
    
    if best_2d_index.shape[0] > 1:
        # Multiple maximum values - randomly select one to break ties
        random_idx = random.randrange(best_2d_index.shape[0])
        best_2d_index = best_2d_index[random_idx, :]
    
    return best_2d_index.squeeze()


def process_state(state, observation_shape):
    """
    Preprocess state observations for neural network input.
    
    This function converts environment states into the tensor format expected
    by neural networks. For image observations, it handles dimension reordering
    to match PyTorch's expected format (channels-first).
    
    Args:
        state: Raw state observation from environment
        observation_shape (tuple): Expected shape of observations
    
    Returns:
        torch.Tensor: Processed state ready for neural network input
    
    Example:
        # For image observation (height, width, channels) -> (batch, channels, height, width)
        state = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])  # (2, 2, 3)
        processed = process_state(state, (2, 2, 3))  # (1, 3, 2, 2)
    
    For Beginners:
    Neural networks are picky about input format. This function converts
    whatever format the environment gives us into what the neural network expects.
    """
    if len(observation_shape) == 3:
        # 3D observation (likely an image: height x width x channels)
        state = torch.tensor(state)
        
        # Reorder dimensions from (H, W, C) to (C, H, W) for PyTorch
        # PyTorch expects channels-first format for convolutional networks
        state = state.transpose(0, 2).transpose(1, 2)
        
        # Convert to float and add batch dimension
        state = state.float().unsqueeze(0)  # Add batch dimension at front
    
    return state

def generate_parameters(mode, domain, debug=False, seed=None, with_expert=None, wandb_project=None):
    os.environ["WANDB_MODE"] = "online"
    os.environ["WANDB_WATCH"]= "false"

    # config parameters
    config_default = yaml.safe_load(open("config/default.yaml", "r"))
    config_domain = yaml.safe_load(open("config/domain/" + domain + ".yaml", "r"))
    
    # Try to load mode-specific config, fallback to ppo if not found
    try:
        config_mode = yaml.safe_load(open("config/mode/" + mode + ".yaml", "r"))
    except FileNotFoundError:
        print(f"Warning: Config file for mode '{mode}' not found, using ppo.yaml")
        config_mode = yaml.safe_load(open("config/mode/ppo.yaml", "r"))

    # override default random seed
    if seed:
        config_default['seed'] = seed

    config_default['experiment_name'] = 'MultiGrid'  # TODO: change me

    # Merge configs
    config_with_domain = merge_configs(config_domain, config_default)
    config = dotdict(merge_configs(config_mode, config_with_domain))

    if debug:
        # Disable weights and biases logging during debugging
        print('Debug selected, disabling wandb')
        wandb.init(project = wandb_project + '-' + domain, config=config,
            mode='disabled')
    else:
        wandb.init(project = wandb_project + '-' + domain, config=config)

    # Get algorithm name from config, fallback to mode
    algorithm_name = getattr(config, 'algorithm', mode)
    
    path_configs = {'model_name': algorithm_name + "_seed_" + str(config.seed) + "_domain_" + config.domain + "_version_" + config.version,
                    'load_model_path': config.get('load_model_start_path', algorithm_name + "_agent_") + "_seed_" + str(config.seed) + "_domain_" + config.domain + "_version_" + config.version,
                    'wandb_project': wandb_project + '-' + config.domain}
    wandb.config.update(path_configs)

    print("CONFIG")
    print(wandb.config)

    wandb.define_metric("episode/x_axis")
    wandb.define_metric("step/x_axis")

    # set all other train/ metrics to use this step
    wandb.define_metric("episode/*", step_metric="episode/x_axis")
    wandb.define_metric("step/*", step_metric="step/x_axis")

    if not os.path.exists("models/"):
        os.makedirs("models/")

    if not os.path.exists("traj/"):
        os.makedirs("traj/")

    wandb.run.name = config.model_name

    return wandb.config


def plot_single_frame(frame_id, full_env_image, agents_partial_images, actions, rewards, action_dict,
                      fig_dir, expt_name, figsize=(10,10), shared_ylim=False, min_ylim=.0001, **kwargs):
    # Seaborn palette.
    sns.set()
    color_palette = sns.palettes.color_palette()

    # Hardcoded plot settings
    linewidth = 1.25
    ms_current = 9
    xlabelpad = 9
    ylabelpad = 10

    # Determine variables
    n_agents = len(actions)
    max_val = np.max(full_env_image)

    # Create figure
    fig = plt.figure(constrained_layout=True, figsize=figsize)
    total_subplots_horizontal = 2 + n_agents
    total_subplots_vertical = 3
    gs = GridSpec(total_subplots_vertical, total_subplots_horizontal, figure=fig)

    # Create sub plots as grid
    full_obs_ax = fig.add_subplot(gs[:2, :2])  # Overall view fig is 2x2 (larger)
    collective_reward_ax = fig.add_subplot(gs[2,:2])
    agents_obs_axes = []
    agents_rewards_axes = []
    for i in range(n_agents):
        agents_obs_axes.append(fig.add_subplot(gs[0, i+2]))
        agents_rewards_axes.append(fig.add_subplot(gs[2, i+2]))

    # Determine grid proportions
    full_obs_proportion = 2.0 / total_subplots_horizontal
    agent_proportion = 1.0 / total_subplots_horizontal

    # Plot shared obervation in top left
    full_obs_ax.imshow(full_env_image, interpolation='none')
    full_obs_ax.set_title('Full environment state')
    full_obs_ax.grid(False)

    # Plot individual agents' observations across top right
    for i in range(n_agents):
        agents_obs_axes[i].imshow(agents_partial_images[i], interpolation='none')
        agents_obs_axes[i].set_title('Agent' + str(i) + ' partial obs')
        agents_obs_axes[i].grid(False)

    # Plot collective return bottom left
    collective_return = np.sum(rewards,axis=1)
    cum_collective_return = np.cumsum(collective_return)
    steps = np.arange(len(cum_collective_return))
    collective_reward_ax.plot(steps, cum_collective_return, color=color_palette[0], lw=linewidth)
    if frame_id > 0:
        collective_reward_ax.plot(frame_id, cum_collective_return[frame_id - 1], 'o', ms=ms_current,
              mfc=color_palette[0], mew=0)

        # Write the reward for previous timestep
        s = 'R_t={}: {}'.format(frame_id-1, collective_return[frame_id-1])
        collective_reward_ax.text(0.1, .85, s, fontsize=10,
                                  horizontalalignment='left', verticalalignment='bottom', transform=collective_reward_ax.transAxes)
    collective_reward_ax.set_xlabel('Step', fontsize=10, labelpad=xlabelpad)
    collective_reward_ax.set_ylabel('Collective return', fontsize=10, labelpad=ylabelpad)

    # Write the reward for current timestep
    s = 'R_t={}: {}'.format(frame_id, collective_return[frame_id])
    collective_reward_ax.text(0.1, 0.7, s, fontsize=10,
                              horizontalalignment='left', verticalalignment='bottom', transform=collective_reward_ax.transAxes)

    # Plot individual agent returns and actions
    for i in range(n_agents):
        # Cumulative return graphs across bottom right
        cum_return = np.cumsum(rewards[:,i])
        agents_rewards_axes[i].plot(steps, cum_return, color=color_palette[0], lw=linewidth)
        if frame_id > 0:
            agents_rewards_axes[i].plot(frame_id, cum_return[frame_id - 1], 'o', ms=ms_current, mfc=color_palette[0], mew=0)
        agents_rewards_axes[i].set_xlabel('Step', fontsize=10, labelpad=xlabelpad)
        agents_rewards_axes[i].set_ylabel('Agent' + str(i) + ' return', fontsize=10, labelpad=ylabelpad)

        # Write the current action and rewards in the space between subplots
        text_horizontal_loc = full_obs_proportion + agent_proportion * i + agent_proportion * 0.2
        if "predicted_actions" in kwargs.keys():
            text_vertical_loc = 0.75
        else:
            text_vertical_loc = 0.65
        act_text = 'a^{}_t={}: {}'.format(i, frame_id, action_dict[int(actions[i])])  # action
        fig.text(text_horizontal_loc, text_vertical_loc, act_text, fontsize=10)
        r_text = 'R_t={}: {}'.format(frame_id, rewards[frame_id, i])
        fig.text(text_horizontal_loc, text_vertical_loc-0.1, r_text, fontsize=10)
        if frame_id > 0:
            r_prev_text = 'R_t={}: {}'.format(frame_id-1, rewards[frame_id-1, i])
            fig.text(text_horizontal_loc, text_vertical_loc-0.05, r_prev_text, fontsize=10)

    filename = '{}_{:05d}.png'.format(expt_name, frame_id)
    fig_path = os.path.join(fig_dir, filename)
    plt.savefig(fig_path)
    plt.close()

def make_video(video_path, video_name='trajectory_video', frame_rate=10, img_extension='.png'):
    image_files = [os.path.join(video_path, img) for img in os.listdir(video_path) if img.endswith(img_extension)]
    image_files.sort()

    clips = [ImageClip(img).set_duration(1) for img in image_files]
    concat_clip = concatenate_videoclips(clips, method="compose")
    concat_clip.write_videofile(os.path.join(video_path, video_name + '.mp4'), fps=frame_rate)

    # Another option: os.system("ffmpeg -r 1 -i img%01d.png -vcodec mpeg4 -y movie.mp4")

def print_network_params(net):
    for name, p in net.named_parameters():
        print(name, p.data.shape)

def extract_mode_from_path(str):
    for mode in ['dqn', 'bcaux', 'basis', 'psiphi', 'copy']:
        if mode in str:
            return mode
    assert False, 'No known mode in path ' + str
