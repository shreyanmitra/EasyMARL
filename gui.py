import gradio as gr
import argparse
import random
import torch
import numpy as np
import wandb

import utils
from multiagent_metacontroller import MultiAgent

AVAILABLE_ENVS = ["MultiGrid-Cluttered-Fixed-15x15"]
currentEnv = None

def getPlot():
    if currentEnv is None:
        return None
    return currentEnv.window.fig

def buttonClicked(env, method):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = utils.generate_parameters(
      mode=method, domain=env, debug=False,
      seed=None, with_expert=None, wandb_project=env)

    currentEnv = utils.make_env(env)

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    agent = MultiAgent(config, env, device)
    agent.train(env)


with gr.Blocks() as interface:
    gr.Markdown("# EasyMarl <br> This is the EasyMARL GUI. We hope to include options for more finegrained training control in the future!")
    with gr.Column():
        env = gr.Dropdown(label="Environment Name", choices = AVAILABLE_ENVS)
        method = gr.Dropdown(label = "Method", choices = ["ppo"])
        train = gr.Button(value = "Train")
    with gr.Column():
        plot = gr.Plot(getPlot, every = 2.0)
        train.click(buttonClicked, inputs = [env, method])

    gr.Markdown("(C) Shreyan Mitra, 2024")

interface.launch(share = True, debug = False)
