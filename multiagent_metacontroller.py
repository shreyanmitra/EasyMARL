from copy import deepcopy
import gym
from itertools import count
import math
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import torch
from torch.distributions.categorical import Categorical
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import wandb

from utils import plot_single_frame, make_video, extract_mode_from_path
from networks.multigrid_network import MultiGridNetwork
from agent import PPOAgent
from envs.gym_multigrid.multigrid import MultiGridEnv

class MultiAgent():
    """This is a meta agent that creates and controls several sub agents. If model_others is True,
    Enables sharing of buffer experience data between agents to allow them to learn models of the
    other agents. """

    def __init__(self, config, env, device, training=True, with_expert=None, debug=False):
        self.training = training
        self.with_expert = with_expert
        self.currentLog = ""
        self.debug = debug
        self.config = config
        self.model_others = not (self.config.mode == "PPO")
        self.agentclass = self.getAgentClass()
        self.device = device
        torch.device(device)
        self.network = MultiGridNetwork
        assert(type(env), MultiGridEnv)
        self.env = env
        self.n_agents = self.env.n_agents
        self.agents = []
        if self.training:
            for i in range(self.n_agents):
                    self.agents.append(self.agentclass(self.env, self.config, self.network, i))
        else:
            self.load_models()

        assert(len(self.agents) == self.n_agents), "Something went wrong when setting up the agents"
        self.total_steps = 0


    def getAgentClass(self):
        if self.config.mode == "PPO":
            return PPOAgent
        else:
            raise NotImplementedError("Selected mode not supported yet")



    def run_one_episode(self, env, episode, log=True, train=True, save_model=True, visualize=False):
        episode_length = 0
        rewards = []
        state = self.env.reset()
        done = False

        if visualize:
            viz_data = {}
            viz_data['actions'] = []
            viz_data['agents_partial_images'] = []
            viz_data['full_images'] = []
            viz_data['predicted_actions'] = []


        while not done:
            if log:
                self.env.render()
            episode_length += 1
            actions = []
            for i in range(self.n_agents):
                currentAgent = self.agents[i]
                actions.append(currentAgent.get_action_predictions()[0])

            next_state, reward, done, _ = self.env.step(actions)

            rewards.append(reward)

            self.currentLog = self.currentLog + "Episode clock: " + str(episode_length) + " Actions: " + str(actions) + " Rewards: " + str(rewards) + " Done? " + str(done) + "\n\n"


            if visualize:
                viz_data = self.add_visualization_data(viz_data, env, state, actions, next_state)

            state = next_state

        for i in range(self.n_agents):
            currentAgent = self.agents[i]
            if "eps_length" not in currentAgent.memory.keys():
                currentAgent.memory["eps_length"] = []
            currentAgent.memory["eps_length"].append(episode_length)
            if "rewards" not in currentAgent.memory.keys():
                currentAgent.memory["rewards"] = []
            currentAgent.memory["rewards"].append(np.array(rewards)[:,i])




        # Logging and checkpointing
        if log: self.log_one_episode(episode, episode_length, rewards)
        self.print_terminal_output(episode, np.sum(rewards))
        #if save_model:
        #self.save_model_checkpoints(episode)

        self.currentLog = ""

        if visualize:
            viz_data['rewards'] = np.array(rewards)
            return viz_data

        if train:
            self.update_models()

        self.total_steps += episode_length

    def log_one_episode(self, episode, episode_length, rewards):
        print(self.currentLog)


    def save_model_checkpoints(self, episode):
        if episode % self.config.save_model_episode == 0:
            for i in range(self.n_agents):
                self.agents[i].save_model()

    def print_terminal_output(self, episode, total_reward):
        if episode % self.config.print_every == 0:
            print('Total steps: {} \t Episode: {} \t Total reward: {}'.format(
                self.total_steps, episode, total_reward))

    def init_visualization_data(self, env, state):
        viz_data = {
            'agents_partial_images': [],
            'actions': [],
            'full_images': [],
            'predicted_actions': None
            }
        viz_data['full_images'].append(env.render('rgb_array'))

        if self.model_others:
            predicted_actions = []
            predicted_actions.append(self.get_action_predictions(state))
            viz_data['predicted_actions'] = predicted_actions

        return viz_data

    def add_visualization_data(self, viz_data, env, state, actions, next_state):
        viz_data['actions'].append(actions)
        viz_data['agents_partial_images'].append(
            [env.get_obs_render(
                self.get_agent_state(state, i)['image']) for i in range(self.n_agents)])
        viz_data['full_images'].append(env.render('rgb_array'))
        if self.model_others:
            viz_data['predicted_actions'].append(self.get_action_predictions(next_state))
        return viz_data

    def get_agent_state(self, state, id):
        return {'image': state['image'][id], 'direction': state['direction'][id]}

    def update_models(self):
        # Don't update model until you've taken enough steps in env
        if self.total_steps > self.config.initial_memory:
            if self.total_steps % self.config.update_every == 0: # How often to update model
                for i in range(self.n_agents):
                    self.agents[i].update()

    def train(self, env):
        if not self.training:
            raise NotImplementedError("Cannot train an agent configured with train = False")
        for episode in range(self.config.n_episodes):
            if episode % self.config.visualize_every == 0 and not (self.debug and episode == 0):
                viz_data = self.run_one_episode(env, episode, visualize=True)
                self.visualize(env, self.config.mode + '_training_step' + str(episode),
                               viz_data=viz_data)
            else:
                self.run_one_episode(env, episode)

        env.close()
        return

    def visualize(self, env, mode, video_dir='videos', viz_data=None):
        if not viz_data:
            viz_data = self.run_one_episode(
                env, episode=0, log=False, train=False, save_model=False, visualize=True)
            env.close()

        video_path = os.path.join(*[video_dir, self.config.experiment_name, self.config.model_name])

        # Set up directory.
        if not os.path.exists(video_path):
            os.makedirs(video_path)

        # Get names of actions
        action_dict = {}
        for act in env.Actions:
            action_dict[act.value] = act.name

        traj_len = len(viz_data['rewards'])
        for t in range(traj_len):
            self.visualize_one_frame(t, viz_data, action_dict, video_path, self.config.model_name)
            print('Frame {}/{}'.format(t, traj_len))

        make_video(video_path, mode + '_trajectory_video')

    def visualize_one_frame(self, t, viz_data, action_dict, video_path, model_name):
        plot_single_frame(t,
                          viz_data['full_images'][t],
                          viz_data['agents_partial_images'][t],
                          viz_data['actions'][t],
                          viz_data['rewards'],
                          action_dict,
                          video_path,
                          self.config.model_name,
                          predicted_actions=viz_data['predicted_actions'],
                          all_actions=viz_data['actions'])

    def load_models(self, model_path=None):
        for i in range(self.n_agents):
            if model_path is not None:
                self.agents[i].actor = torch.load(model_path + '_agent_actor_' + str(i))
                self.agents[i].critic = torch.load(model_path + '_agent_critic_' + str(i))
            else:
                # Use agents' default model path
                self.agents[i].actor = torch.load(self.config.mode + '_agent_actor_' + str(i))
                self.agents[i].critic = torch.load(self.config.mode + '_agent_critic_' + str(i))
